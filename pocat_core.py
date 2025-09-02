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
                    derated_limit = calculate_derated_current_limit(concrete_ic, constraints)
                    if derated_limit <= 0: continue
                    concrete_ic.i_limit = derated_limit
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

    margin = constraints.get('current_margin', 0.1)
    for p in candidate_ics:
        terms = [child_current_draw[c.name] * edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        if terms: model.Add(sum(terms) <= int(p.i_limit * (1 + margin) * SCALE))
        
    if 'power_sequences' in constraints:
        for seq in constraints['power_sequences']:
            if seq.get('f') == 1:
                j, k = seq['j'], seq['k'];
                for p in candidate_ics:
                    if (p.name, j) in edges and (p.name, k) in edges: model.Add(edges[p.name, j] + edges[p.name, k] <= 1)

    # --- [핵심 수정] "soft"와 "hard" Independent Rail 제약조건 로직 (최종 수정) ---
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
            
            # 1. hard load 자신은 항상 '전용도로' 위에 있음
            model.Add(is_on_hard_path[load.name] == 1)
            for other_load in loads:
                if other_load.name != load.name:
                    model.Add(is_on_hard_path[other_load.name] == 0)

            # 2. '전용도로' 속성을 부모에게 연쇄적으로 전파
            for c_node in all_ic_and_load_nodes:
                for p_node in parent_nodes:
                    if (p_node.name, c_node.name) in edges:
                        # "만약 (부모 p -> 자식 c) 연결이 활성화되고, 자식 c가 경로 위에 있다면, 부모 p도 경로 위에 있어야 한다."
                        model.AddImplication(is_on_hard_path[c_node.name], is_on_hard_path[p_node.name]).OnlyEnforceIf(edges[(p_node.name, c_node.name)])
            
            # 3. 최종 규칙: '전용도로' 위의 모든 IC는 자식을 하나만 가질 수 있음
            for p_ic in candidate_ics:
                model.Add(num_children_vars[p_ic.name] <= 1).OnlyEnforceIf(is_on_hard_path[p_ic.name])
    # --- 수정 끝 ---
    
    if constraints.get('max_sleep_current', 0) > 0:
        ic_sleep_terms = [int(ic.operating_current * SCALE) * ic_is_used[ic.name] for ic in candidate_ics]
        always_on_loads = [l for l in loads if l.always_on_in_sleep]
        load_sleep_total = int(sum(load.current_sleep for load in always_on_loads) * SCALE)
        model.Add(sum(ic_sleep_terms) + load_sleep_total <= int(constraints['max_sleep_current'] * SCALE))
        
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