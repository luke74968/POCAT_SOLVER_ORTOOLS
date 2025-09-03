# pocat_core.py
import json
import copy
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from ortools.sat.python import cp_model

from pocat_classes import Battery, Load, PowerIC, LDO, BuckConverter
# 순환 참조를 피하기 위해 함수를 직접 임포트하지 않고, main에서 넘겨받도록 구조 변경
# from pocat_visualizer import check_solution_validity, print_and_visualize_one_solution

# 솔버 콜백 클래스
class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, ic_is_used, edges):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.solutions = []
    def on_solution_callback(self):
        self.__solution_count += 1
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]}
        self.solutions.append(current_solution)
    def solution_count(self): return self.__solution_count

class SolutionLogger(cp_model.CpSolverSolutionCallback):
    def __init__(self, ic_is_used, edges, limit=1):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.limit = limit
        self.solutions = []
    def on_solution_callback(self):
        if len(self.solutions) >= self.limit:
            self.StopSearch()
            return
        self.__solution_count += 1
        print(f"  -> 대표 솔루션 #{self.__solution_count} 발견!")
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]
        }
        self.solutions.append(current_solution)

# 핵심 로직 함수들
def calculate_derated_current_limit(ic: PowerIC, constraints: Dict[str, Any]) -> float:
    ambient_temp = constraints.get('ambient_temperature', 25)
    thermal_margin_percent = constraints.get('thermal_margin_percent', 0)
    if ic.theta_ja == 0: return ic.i_limit
    temp_rise_allowed = ic.t_junction_max - ambient_temp
    if temp_rise_allowed <= 0: return 0
    p_loss_max = (temp_rise_allowed / (ic.theta_ja * (1 + thermal_margin_percent)))
    i_limit_based_temp = ic.i_limit
    if isinstance(ic, LDO):
        vin, vout = ic.vin, ic.vout; op_current = ic.operating_current
        numerator = p_loss_max - (vin * op_current); denominator = vin - vout
        if denominator > 0 and numerator > 0: i_limit_based_temp = numerator / denominator
    elif isinstance(ic, BuckConverter):
        current_check = ic.i_limit
        while current_check > 0:
            if ic.calculate_power_loss(ic.vin, current_check) <= p_loss_max:
                i_limit_based_temp = current_check; break
            current_check -= 0.001
        else: i_limit_based_temp = 0
    return min(ic.i_limit, i_limit_based_temp)

def load_configuration(config_string: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    config = json.loads(config_string); battery = Battery(**config['battery']); available_ics = []
    for ic_data in config['available_ics']:
        ic_type = ic_data.pop('type')
        if ic_type == 'LDO': available_ics.append(LDO(**ic_data))
        elif ic_type == 'Buck': available_ics.append(BuckConverter(**ic_data))
    loads = [Load(**load_data) for load_data in config['loads']]; constraints = config['constraints']
    print("✅ 설정 파일 로딩 완료!")
    return battery, available_ics, loads, constraints

def expand_ic_instances(available_ics: List[PowerIC], loads: List[Load], battery: Battery, constraints: Dict[str, Any]) -> Tuple[List[PowerIC], Dict[str, List[str]]]:
    print("\n⚙️  IC 인스턴스 확장 및 복제 시작...")
    potential_vout = sorted(list(set(load.voltage_typical for load in loads)))
    battery.vout = (battery.voltage_min + battery.voltage_max) / 2
    potential_vin = sorted(list(set([battery.vout] + potential_vout))); candidate_ics, ic_groups = [], {}
    for template_ic in available_ics:
        for vin in potential_vin:
            for vout in potential_vout:
                if not (template_ic.vin_min <= vin <= template_ic.vin_max): continue
                if not (template_ic.vout_min <= vout <= template_ic.vout_max): continue
                if isinstance(template_ic, LDO):
                    if vin < (vout + template_ic.v_dropout): continue
                elif isinstance(template_ic, BuckConverter):
                    if vin <= vout: continue
                num_potential_loads = sum(1 for load in loads if load.voltage_typical == vout)
                group_key = f"{template_ic.name}@{vin:.1f}Vin_{vout:.1f}Vout"; current_group = []
                for i in range(num_potential_loads):
                    concrete_ic = copy.deepcopy(template_ic); concrete_ic.vin, concrete_ic.vout = vin, vout
                    concrete_ic.name = f"{group_key}_copy{i+1}"
                    
                    # --- [핵심 수정] 열 마진 계산 전, 원래 스펙 저장 ---
                    concrete_ic.original_i_limit = template_ic.i_limit
                    # --- 수정 끝 ---

                    derated_limit = calculate_derated_current_limit(concrete_ic, constraints)
                    if derated_limit <= 0: continue
                    concrete_ic.i_limit = derated_limit # 열 마진이 적용된 값으로 덮어쓰기
                    candidate_ics.append(concrete_ic); current_group.append(concrete_ic.name)
                if current_group: ic_groups[group_key] = current_group
    print(f"   - (필터링 포함) 생성된 최종 후보 IC 인스턴스: {len(candidate_ics)}개")
    return candidate_ics, ic_groups

def _initialize_model_variables(model, candidate_ics, loads, battery):
    """모델의 기본 변수들(노드, 엣지, IC 사용 여부)을 생성하고 반환합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics
    all_nodes = parent_nodes + all_ic_and_load_nodes
    node_names = list(set(n.name for n in all_nodes))
    ic_names = [ic.name for ic in candidate_ics]
    
    edges = {}
    for p in parent_nodes:
        for c in all_ic_and_load_nodes:
            if p.name == c.name: continue
            is_compatible = False
            if p.name == battery.name:
                if isinstance(c, PowerIC) and (c.vin_min <= battery.voltage_min and battery.voltage_max <= c.vin_max):
                    is_compatible = True
            elif isinstance(p, PowerIC):
                child_vin_req = c.vin if hasattr(c, 'vin') else c.voltage_typical
                if p.vout == child_vin_req:
                    is_compatible = True
            if is_compatible:
                edges[(p.name, c.name)] = model.NewBoolVar(f'edge_{p.name}_to_{c.name}')
    
    ic_is_used = {ic.name: model.NewBoolVar(f'is_used_{ic.name}') for ic in candidate_ics}
    
    print(f"   - (필터링 후) 생성된 'edge' 변수: {len(edges)}개")
    # `parent_nodes`를 반환 값에 추가
    return all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used

# --- 💡 2. 각 제약 조건을 추가하는 함수들 ---
def add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used):
    """전력망의 가장 기본적인 연결 규칙을 정의합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics

    # 사용되는 IC는 반드시 출력이 있어야 함
    for ic in candidate_ics:
        outgoing = [edges[ic.name, c.name] for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if outgoing:
            model.Add(sum(outgoing) > 0).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(outgoing) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())
        else:
            model.Add(ic_is_used[ic.name] == False)
    # 모든 부하는 반드시 하나의 부모를 가져야 함
    for load in loads:
        possible_parents = [edges[p.name, load.name] for p in parent_nodes if (p.name, load.name) in edges]
        if possible_parents: model.AddExactlyOne(possible_parents)
    # 사용되는 IC는 반드시 하나의 부모를 가져야 함
    for ic in candidate_ics:
        incoming = [edges[p.name, ic.name] for p in parent_nodes if (p.name, ic.name) in edges]
        if incoming:
            model.Add(sum(incoming) == 1).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(incoming) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())

def add_ic_group_constraints(model, ic_groups, ic_is_used):
    """복제된 IC 그룹 내에서의 사용 순서를 강제합니다."""
    for copies in ic_groups.values():
        for i in range(len(copies) - 1):
            model.AddImplication(ic_is_used[copies[i+1]], ic_is_used[copies[i]])

def add_current_limit_constraints(model, candidate_ics, loads, constraints, edges):
    """IC의 전류 한계(열 마진, 전기 마진) 제약 조건을 추가합니다."""
    SCALE = 1_000_000
    all_ic_and_load_nodes = candidate_ics + loads
    
    child_current_draw = {node.name: int(node.current_active * SCALE) for node in loads}
    potential_loads_for_ic = defaultdict(list)
    for ic in candidate_ics:
        for load in loads:
            if ic.vout == load.voltage_typical:
                potential_loads_for_ic[ic.name].append(load.current_active)
    for ic in candidate_ics:
        max_potential_i_out = sum(potential_loads_for_ic[ic.name])
        realistic_i_out = min(ic.i_limit, max_potential_i_out)
        child_current_draw[ic.name] = int(ic.calculate_input_current(vin=ic.vin, i_out=realistic_i_out) * SCALE)

    current_margin = constraints.get('current_margin', 0.1)
    for p in candidate_ics:
        terms = [child_current_draw[c.name] * edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        if terms:
            model.Add(sum(terms) <= int(p.i_limit * SCALE))
            model.Add(sum(terms) <= int(p.original_i_limit * (1 - current_margin) * SCALE))

def add_power_sequence_constraints(model, candidate_ics, loads, constraints, node_names, ic_names, edges):
    """전원 시퀀스(동일 부모 금지, 시간적 선후 관계) 제약 조건을 추가합니다."""
    if 'power_sequences' not in constraints or not constraints['power_sequences']:
        return
        
    is_ancestor = {
        (p, c): model.NewBoolVar(f'anc_{p}_to_{c}')
        for p in node_names for c in node_names if p != c
    }
    for p, c in edges:
        model.AddImplication(edges[p, c], is_ancestor[p, c])
    for a in node_names:
        for b in ic_names:
            for c in node_names:
                if a == b or b == c or a == c: continue
                model.AddBoolOr([is_ancestor[a, b].Not(), is_ancestor[b, c].Not(), is_ancestor[a, c]])
    
    parent_ic_vars = defaultdict(list)
    for load in loads:
        for p_ic in candidate_ics:
            if (p_ic.name, load.name) in edges:
                parent_ic_vars[load.name].append((p_ic.name, edges[p_ic.name, load.name]))

    for seq in constraints['power_sequences']:
        if seq.get('f') != 1: continue
        j_name, k_name = seq['j'], seq['k']
        for p in candidate_ics:
            if (p.name, j_name) in edges and (p.name, k_name) in edges:
                model.Add(edges[p.name, j_name] + edges[p.name, k_name] <= 1)
        for p_j_name, j_edge in parent_ic_vars[j_name]:
            for p_k_name, k_edge in parent_ic_vars[k_name]:
                if p_j_name == p_k_name: continue
                model.Add(is_ancestor[p_k_name, p_j_name] == 0).OnlyEnforceIf([j_edge, k_edge])

# --- 💡 3. 재구성된 메인 모델 생성 함수 수정 ---
def create_solver_model(candidate_ics, loads, battery, constraints, ic_groups):
    """
    OR-Tools 모델을 생성하고 모든 제약 조건을 추가한 뒤 반환합니다.
    """
    print("\n🧠 OR-Tools 모델 생성 시작...")
    model = cp_model.CpModel()

    # 1. 변수 초기화
    # `parent_nodes`를 변수로 받음
    all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used = _initialize_model_variables(
        model, candidate_ics, loads, battery
    )
    
    # 2. 제약 조건 추가
    add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used)
    add_ic_group_constraints(model, ic_groups, ic_is_used)
    add_current_limit_constraints(model, candidate_ics, loads, constraints, edges)
    add_power_sequence_constraints(model, candidate_ics, loads, constraints, node_names, ic_names, edges)
    
    # `parent_nodes`를 올바르게 전달
    add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges)

    is_always_on_path = add_always_on_constraints(model, all_nodes, loads, candidate_ics, edges)
    add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path)

    # N. 목표 함수 설정
    cost_objective = sum(int(ic.cost * 10000) * ic_is_used[ic.name] for ic in candidate_ics)
    model.Minimize(cost_objective)
    
    print("✅ 모델 생성 완료!")
    return model, edges, ic_is_used
# --- 💡 Independent Rail 제약조건 함수 ---
def add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges):
    all_ic_and_load_nodes = candidate_ics + loads
    num_children_vars = {p.name: model.NewIntVar(0, len(all_ic_and_load_nodes), f"num_children_{p.name}") for p in parent_nodes}
    for p in parent_nodes:
        outgoing_edges = [edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        model.Add(num_children_vars[p.name] == sum(outgoing_edges))

    for load in loads:
        if load.independent_rail_type == 'soft':
            for p_ic in candidate_ics:
                if (p_ic.name, load.name) in edges:
                    model.Add(num_children_vars[p_ic.name] == 1).OnlyEnforceIf(edges[(p_ic.name, load.name)])
        
        elif load.independent_rail_type == 'hard':
            is_on_hard_path = {node.name: model.NewBoolVar(f"on_hard_path_{load.name}_{node.name}") for node in all_nodes}
            model.Add(is_on_hard_path[load.name] == 1)
            for other_load in loads:
                if other_load.name != load.name:
                    model.Add(is_on_hard_path[other_load.name] == 0)
            for c_node in all_ic_and_load_nodes:
                for p_node in parent_nodes:
                    if (p_node.name, c_node.name) in edges:
                        model.AddImplication(is_on_hard_path[c_node.name], is_on_hard_path[p_node.name]).OnlyEnforceIf(edges[(p_node.name, c_node.name)])
            for p_ic in candidate_ics:
                model.Add(num_children_vars[p_ic.name] <= 1).OnlyEnforceIf(is_on_hard_path[p_ic.name])


# --- 💡 Always-On 및 Sleep Current 제약조건 함수 ---
def add_always_on_constraints(model, all_nodes, loads, candidate_ics, edges):
    all_ic_and_load_nodes = candidate_ics + loads
    is_always_on_path = {node.name: model.NewBoolVar(f"is_ao_{node.name}") for node in all_nodes}
    for ld in loads:
        model.Add(is_always_on_path[ld.name] == int(ld.always_on_in_sleep))
    for ic in candidate_ics:
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if not children:
            model.Add(is_always_on_path[ic.name] == 0)
            continue
        z_list = []
        for ch in children:
            e = edges[(ic.name, ch.name)]
            z = model.NewBoolVar(f"ao_and_{ic.name}__{ch.name}")
            model.Add(z <= e); model.Add(z <= is_always_on_path[ch.name]); model.Add(z >= e + is_always_on_path[ch.name] - 1)
            z_list.append(z)
        for z in z_list: model.Add(is_always_on_path[ic.name] >= z)
        model.Add(is_always_on_path[ic.name] <= sum(z_list))
    for p in candidate_ics:
        chs = [c for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        for i in range(len(chs) - 1):
            for j in range(i + 1, len(chs)):
                c1, c2 = chs[i], chs[j]
                model.Add(is_always_on_path[c1.name] == is_always_on_path[c2.name]).OnlyEnforceIf([edges[(p.name, c1.name)], edges[(p.name, c2.name)]])
    return is_always_on_path


def add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path):
    SCALE = 1_000_000
    if constraints.get('max_sleep_current', 0) > 0:
        sleep_terms = []
        for ic in candidate_ics:
            ao = is_always_on_path[ic.name]
            ao_term = int(ic.operating_current * SCALE) * ao
            top_edge = edges.get((battery.name, ic.name), None)
            if top_edge is not None:
                not_ao = model.NewBoolVar(f"not_ao_{ic.name}"); model.Add(not_ao + ao == 1)
                z_top_non_ao = model.NewBoolVar(f"top_non_ao_{ic.name}")
                model.Add(z_top_non_ao <= top_edge); model.Add(z_top_non_ao <= not_ao); model.Add(z_top_non_ao >= top_edge + not_ao - 1)
                iq_term = int(ic.quiescent_current * SCALE) * z_top_non_ao
                sleep_terms.append(ao_term + iq_term)
            else:
                sleep_terms.append(ao_term)
        load_sleep_total = int(sum(l.current_sleep for l in loads if l.always_on_in_sleep) * SCALE)
        model.Add(sum(sleep_terms) + load_sleep_total <= int(constraints['max_sleep_current'] * SCALE))

# 원본의 병렬해 탐색 함수
def find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints, viz_func, check_func):
    print("\n\n👑 --- 최종 단계: 모든 부하 분배 조합 탐색 --- 👑")
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    ic_type_to_instances = defaultdict(list)
    for ic_name in base_solution['used_ic_names']:
        ic = candidate_ics_map[ic_name]
        ic_type = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
        ic_type_to_instances[ic_type].append(ic)

    instance_to_children = defaultdict(set)
    for p, c in base_solution['active_edges']:
        if p in candidate_ics_map:
            instance_to_children[p].add(c)
    
    target_group = None
    for ic_type, instances in ic_type_to_instances.items():
        if len(instances) > 1:
            total_load_pool = set()
            for inst in instances:
                total_load_pool.update(instance_to_children[inst.name])
            target_group = {
                'instances': [inst.name for inst in instances],
                'load_pool': list(total_load_pool)
            }
            break

    if not target_group:
        print("\n -> 이 해답에는 생성할 병렬해가 없습니다.")
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
             viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    def find_partitions(items, num_bins):
        if not items:
            yield [[] for _ in range(num_bins)]
        else:
            for partition in find_partitions(items[1:], num_bins):
                for i in range(num_bins):
                    yield partition[:i] + [[items[0]] + partition[i]] + partition[i+1:]
                if num_bins > len(partition):
                    yield partition + [[items[0]]]

    valid_solutions = []
    seen_partitions = set()
    num_instances = len(target_group['instances'])
    load_pool = target_group['load_pool']

    for p in find_partitions(load_pool, num_instances):
        if len(p) == num_instances:
            canonical_partition = tuple(sorted([tuple(sorted(sublist)) for sublist in p]))
            if canonical_partition in seen_partitions:
                continue
            seen_partitions.add(canonical_partition)
            new_edges = [edge for edge in base_solution['active_edges'] if edge[0] not in target_group['instances']]
            for i, instance_name in enumerate(target_group['instances']):
                for load_name in p[i]:
                    new_edges.append((instance_name, load_name))
            new_solution = {"used_ic_names": base_solution['used_ic_names'], "active_edges": new_edges, "cost": base_solution['cost']}
            if check_func(new_solution, candidate_ics, loads, battery, constraints):
                valid_solutions.append(new_solution)
    
    print(f"\n✅ 총 {len(valid_solutions)}개의 유효한 병렬해 구조를 찾았습니다.")
    for i, solution in enumerate(valid_solutions):
        print(f"\n--- [병렬해 #{i+1}] ---")
        viz_func(solution, candidate_ics, loads, battery, constraints, solution_index=i+1)