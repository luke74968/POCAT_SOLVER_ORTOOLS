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

def create_solver_model(candidate_ics, loads, battery, constraints, ic_groups):
    print("\n🧠 OR-Tools 모델 생성 시작...")
    model = cp_model.CpModel()
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics
    all_nodes = parent_nodes + loads
    node_names = [n.name for n in all_nodes]
    ic_names = [ic.name for ic in candidate_ics]
    edges = {}
    
    for p in parent_nodes:
        for c in all_ic_and_load_nodes:
            if p.name == c.name: continue
            is_compatible = False
            if p.name == battery.name:
                if isinstance(c, PowerIC) and (c.vin_min <= p.voltage_min and p.voltage_max <= c.vin_max): is_compatible = True
            else:
                child_vin_req = c.vin if hasattr(c, 'vin') else c.voltage_typical
                if p.vout == child_vin_req: is_compatible = True
            if is_compatible: edges[(p.name, c.name)] = model.NewBoolVar(f'edge_{p.name}_to_{c.name}')
    print(f"   - (필터링 후) 생성된 'edge' 변수: {len(edges)}개")

    ic_is_used = {ic.name: model.NewBoolVar(f'is_used_{ic.name}') for ic in candidate_ics}
    for ic in candidate_ics:
        outgoing = [edges[ic.name, c.name] for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if outgoing:
            model.Add(sum(outgoing) > 0).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(outgoing) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())
        else: model.Add(ic_is_used[ic.name] == False)

    for load in loads:
        possible_parents = [edges[p.name, load.name] for p in parent_nodes if (p.name, load.name) in edges]
        if possible_parents: model.AddExactlyOne(possible_parents)

    for ic in candidate_ics:
        incoming = [edges[p.name, ic.name] for p in parent_nodes if (p.name, ic.name) in edges]
        if incoming:
            model.Add(sum(incoming) == 1).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(incoming) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())
            
    for copies in ic_groups.values():
        for i in range(len(copies) - 1): model.AddImplication(ic_is_used[copies[i+1]], ic_is_used[copies[i]])
        
    SCALE = 1_000_000
    child_current_draw = {node.name: int(node.current_active * SCALE) for node in loads}
    potential_loads_for_ic = defaultdict(list)
    for ic in candidate_ics:
        for load in loads:
            if ic.vout == load.voltage_typical: potential_loads_for_ic[ic.name].append(load.current_active)
    for ic in candidate_ics:
        max_potential_i_out = sum(potential_loads_for_ic[ic.name])
        realistic_i_out = min(ic.i_limit, max_potential_i_out)
        child_current_draw[ic.name] = int(ic.calculate_input_current(vin=ic.vin, i_out=realistic_i_out) * SCALE)

    # --- [핵심 수정] 열 마진과 전기 마진 제약조건 분리 ---
    current_margin = constraints.get('current_margin', 0.1)
    for p in candidate_ics:
        terms = [child_current_draw[c.name] * edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        if terms:
            # 1. 열 마진이 적용된 전류 한계 (절대 넘으면 안되는 값)
            # p.i_limit은 이미 derating된 값임
            model.Add(sum(terms) <= int(p.i_limit * SCALE))
            
            # 2. 전기 마진이 적용된 전류 한계 (설계 여유분)
            # p.original_i_limit은 derating 전의 원래 스펙
            model.Add(sum(terms) <= int(p.original_i_limit * (1 - current_margin) * SCALE))

        # --- 💡 로직 개선: 전원 시퀀스 제약 강화 ---
    if 'power_sequences' in constraints and constraints['power_sequences']:
        # 1. 경로 추적을 위한 is_ancestor 변수 생성
        is_ancestor = {
            (p, c): model.NewBoolVar(f'anc_{p}_to_{c}')
            for p in node_names for c in node_names if p != c
        }
        # 2. 경로 제약 조건 설정 (A->B이고 B->C이면, A->C이다)
        for p, c in edges: # 직접 연결은 조상 관계
            model.AddImplication(edges[p, c], is_ancestor[p, c])
        for a in node_names: # 간접 연결(경로)도 조상 관계 (Transitive Closure)
            for b in ic_names:
                for c in node_names:
                    if a == b or b == c or a == c: continue
                    # (a->b)이고 (b->c)이면 (a->c)가 참이 되어야 함
                    model.AddBoolOr([
                        is_ancestor[a, b].Not(),
                        is_ancestor[b, c].Not(),
                        is_ancestor[a, c]
                    ])
        
        # 3. 강화된 시퀀스 제약 적용
        parent_ic_vars = defaultdict(list)
        for load in loads:
            for p_ic in candidate_ics:
                if (p_ic.name, load.name) in edges:
                    parent_ic_vars[load.name].append((p_ic.name, edges[p_ic.name, load.name]))

        for seq in constraints['power_sequences']:
            if seq.get('f') != 1: continue
            j_name, k_name = seq['j'], seq['k'] # j가 k보다 먼저 켜져야 함

            # 기존 제약 (같은 부모 금지)은 여전히 유효
            for p in candidate_ics:
                if (p.name, j_name) in edges and (p.name, k_name) in edges:
                    model.Add(edges[p.name, j_name] + edges[p.name, k_name] <= 1)
            
            # 새로운 제약 (시간적 선후 관계)
            # "k의 부모 IC"는 "j의 부모 IC"의 자손이 될 수 없다.
            # 즉, j_parent는 k_parent의 조상이 될 수 없다.
            for p_j_name, j_edge in parent_ic_vars[j_name]:
                for p_k_name, k_edge in parent_ic_vars[k_name]:
                    if p_j_name == p_k_name: continue # 이미 위에서 처리됨
                    
                    # is_ancestor[k_parent, j_parent]가 참이 되는 것을 방지
                    model.Add(is_ancestor[p_k_name, p_j_name] == 0).OnlyEnforceIf([j_edge, k_edge])

    # --- 수정 끝 ---
        
    if 'power_sequences' in constraints:
        for seq in constraints['power_sequences']:
            if seq.get('f') == 1:
                j, k = seq['j'], seq['k'];
                for p in candidate_ics:
                    if (p.name, j) in edges and (p.name, k) in edges: model.Add(edges[p.name, j] + edges[p.name, k] <= 1)

    # ... (이하 코드는 이전과 동일)
    
    # --- Independent Rail 제약조건 ---
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

    # --- Always-On 경로 분리 + 전파 (정확한 AND/OR 선형화) ---
    # 1) 노드 AO 변수
    is_always_on_path = {node.name: model.NewBoolVar(f"is_ao_{node.name}") for node in all_nodes}

    # 2) Load는 config대로 고정
    for ld in loads:
        model.Add(is_always_on_path[ld.name] == int(ld.always_on_in_sleep))

    # 3) IC의 AO = OR_j ( edge(ic->child_j) AND is_ao[child_j] )
    for ic in candidate_ics:
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if not children:
            model.Add(is_always_on_path[ic.name] == 0)
            continue

        z_list = []  # z_j = edge AND child_is_ao
        for ch in children:
            e = edges[(ic.name, ch.name)]
            z = model.NewBoolVar(f"ao_and_{ic.name}__{ch.name}")
            # z = e ∧ is_ao[ch]  (⇔로 선형화)
            model.Add(z <= e)
            model.Add(z <= is_always_on_path[ch.name])
            model.Add(z >= e + is_always_on_path[ch.name] - 1)
            z_list.append(z)

        # is_ao[ic] == OR(z_list)
        for z in z_list:
            model.Add(is_always_on_path[ic.name] >= z)
        model.Add(is_always_on_path[ic.name] <= sum(z_list))

    # 4) 같은 부모 아래에서 AO/비-AO 혼용 방지(선택)
    for p in candidate_ics:
        chs = [c for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        for i in range(len(chs) - 1):
            for j in range(i + 1, len(chs)):
                c1, c2 = chs[i], chs[j]
                model.Add(is_always_on_path[c1.name] == is_always_on_path[c2.name]).OnlyEnforceIf([
                    edges[(p.name, c1.name)],
                    edges[(p.name, c2.name)],
                ])
    
    # --- Sleep-current budget: AO 경로만 Iop, 비-AO 탑레벨은 Iq, 그 외 0 ---
    if constraints.get('max_sleep_current', 0) > 0:
        sleep_terms = []

        for ic in candidate_ics:
            # AO 여부 (create_solver_model에서 이미 만든 BoolVar)
            ao = is_always_on_path[ic.name]

            # AO면 Iop 포함
            ao_term = int(ic.operating_current * SCALE) * ao

            # 비-AO 탑레벨(배터리 직결) 판정: z_top_non_ao = (battery→ic) AND (NOT ao)
            top_edge = edges.get((battery.name, ic.name), None)
            if top_edge is not None:
                not_ao = model.NewBoolVar(f"not_ao_{ic.name}")
                model.Add(not_ao + ao == 1)  # not_ao = 1 - ao

                z_top_non_ao = model.NewBoolVar(f"top_non_ao_{ic.name}")
                # z = top_edge ∧ not_ao (표준 선형화)
                model.Add(z_top_non_ao <= top_edge)
                model.Add(z_top_non_ao <= not_ao)
                model.Add(z_top_non_ao >= top_edge + not_ao - 1)

                iq_term = int(ic.quiescent_current * SCALE) * z_top_non_ao
                sleep_terms.append(ao_term + iq_term)
            else:
                # 탑레벨이 아닌 비-AO는 0, AO면 ao_term만
                sleep_terms.append(ao_term)

        # AO 로드의 슬립 부하만 합산
        load_sleep_total = int(sum(l.current_sleep for l in loads if l.always_on_in_sleep) * SCALE)

        model.Add(sum(sleep_terms) + load_sleep_total <= int(constraints['max_sleep_current'] * SCALE))






    cost_objective = sum(int(ic.cost * 10000) * ic_is_used[ic.name] for ic in candidate_ics)
    model.Minimize(cost_objective)
    
    print("✅ 모델 생성 완료!")
    return model, edges, ic_is_used

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