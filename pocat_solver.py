# ================================================================
# POCAT: Power Tree Optimization via Constraint-Aware Solver (VS Code Version)
# V3, 최적 단일해 찾은 후 해당 단일해의 대칭해 탐색
# ================================================================

# 1. 라이브러리 임포트
# 로컬 환경에서는 터미널에서 라이브러리를 미리 설치해야 합니다.
# (예: pip install -r requirements.txt)
import json
import copy
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from ortools.sat.python import cp_model
from graphviz import Digraph
from collections import defaultdict
from itertools import permutations

# ================================================================
# 2. 데이터 클래스 정의
# ================================================================
@dataclass
class Battery:
    name: str; voltage_min: float; voltage_max: float; capacity_mah: int; vout: float = 0.0

@dataclass
class Load:
    name: str; voltage_req_min: float; voltage_req_max: float; voltage_typical: float
    current_active: float; current_sleep: float
    independent_rail_type: Optional[str] = None
    always_on_in_sleep: bool = False

@dataclass
class PowerIC:
    name: str; vin_min: float; vin_max: float; vout_min: float; vout_max: float; i_limit: float
    operating_current: float; quiescent_current: float; cost: float; theta_ja: float; t_junction_max: int
    load_dump_rating_v: float = 0.0; vin: float = 0.0; vout: float = 0.0
    def calculate_power_loss(self, vin: float, i_out: float) -> float: raise NotImplementedError
    def calculate_input_current(self, vin: float, i_out: float) -> float: raise NotImplementedError

@dataclass
class LDO(PowerIC):
    type: str = "LDO"; v_dropout: float = 0.0
    def calculate_power_loss(self, vin: float, i_out: float) -> float: return ((vin - self.vout) * i_out) + (vin * self.operating_current)
    def calculate_input_current(self, vin: float, i_out: float) -> float: return i_out + self.operating_current

@dataclass
class BuckConverter(PowerIC):
    type: str = "Buck"; efficiency: Dict[float, float] = field(default_factory=dict)
    def get_efficiency(self, i_out: float) -> float:
        if not self.efficiency or i_out <= 0: return 0.9
        currents = sorted(self.efficiency.keys()); efficiencies = [self.efficiency[c] for c in currents]
        return np.interp(i_out, currents, efficiencies)
    def calculate_power_loss(self, vin: float, i_out: float) -> float:
        p_out = self.vout * i_out; eff = self.get_efficiency(i_out)
        if eff == 0: return float('inf')
        return (p_out / eff) - p_out
    def calculate_input_current(self, vin: float, i_out: float) -> float:
        if vin == 0: return float('inf')
        p_out = self.vout * i_out; eff = self.get_efficiency(i_out)
        if eff == 0: return float('inf')
        p_in = p_out / eff
        return (p_in / vin) + self.operating_current
        
# ================================================================
# 3. 솔버 콜백 클래스 정의
# ================================================================
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
    """솔버가 해답을 찾을 때마다 중간 과정을 저장하고 출력하는 클래스"""
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

# ================================================================
# 4. 설정 (JSON)
# ================================================================
json_config_string = """
{
  "battery": { "name": "Li-Ion_4S", "voltage_min": 12.0, "voltage_max": 16.8, "capacity_mah": 5000 },
  "available_ics": [
    { "type": "Buck", "name": "DCDC_A", "vin_min": 6.0, "vin_max": 18.0, "vout_min": 1.0, "vout_max": 5.5, "i_limit": 2.0, "operating_current": 0.0025, "quiescent_current": 0.0015, "cost": 0.6, "theta_ja": 40.0, "t_junction_max": 150 },
    { "type": "Buck", "name": "DCDC_B", "vin_min": 4.5, "vin_max": 6.0, "vout_min": 1.0, "vout_max": 3.3, "i_limit": 1.5, "operating_current": 0.0020, "quiescent_current": 0.001, "cost": 0.55, "theta_ja": 45.0, "t_junction_max": 150 },
    { "type": "LDO", "name": "LDO_X", "vin_min": 2.0, "vin_max": 6.0, "vout_min": 1.2, "vout_max": 3.3, "i_limit": 0.3, "v_dropout": 0.25, "operating_current": 0.0008, "quiescent_current": 0.00009, "cost": 0.3, "theta_ja": 60.0, "t_junction_max": 125 }
  ],
  "loads": [
    { "name": "MCU_Main", "voltage_req_min": 3.135, "voltage_req_max": 3.465, "voltage_typical": 3.3, "current_active": 0.150, "current_sleep": 0.0001 },
    { "name": "Sensor_Bank", "voltage_req_min": 3.135, "voltage_req_max": 3.465, "voltage_typical": 3.3, "current_active": 0.060, "current_sleep": 0.00005 },
    { "name": "CAN_Transceiver", "voltage_req_min": 4.75, "voltage_req_max": 5.25, "voltage_typical": 5.0, "current_active": 0.070, "current_sleep": 0.0001, "independent_rail_type": "soft" },
    { "name": "RF_Frontend", "voltage_req_min": 3.135, "voltage_req_max": 3.465, "voltage_typical": 3.3, "current_active": 0.300, "current_sleep": 0, "independent_rail_type": "soft" },
    { "name": "DDR_Core", "voltage_req_min": 3.135, "voltage_req_max": 3.465, "voltage_typical": 3.3, "current_active": 0.500, "current_sleep": 0 },
    { "name": "RTC_AlwaysOn", "voltage_req_min": 1.71, "voltage_req_max": 1.89, "voltage_typical": 1.8, "current_active": 0.0005, "current_sleep": 0.00005, "independent_rail_type": "soft", "always_on_in_sleep": true }
  ],
  "constraints": {
    "ambient_temperature": 50,
    "current_margin": 0.25,
    "power_sequences": [{ "j": "MCU_Main", "k": "DDR_Core", "f": 1 }],
    "thermal_margin_percent": 0.15,
    "max_sleep_current": 0.012
  }
}
"""

# ================================================================
# 5. 핵심 로직 함수들
# ================================================================
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
    model = cp_model.CpModel(); all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics; edges = {}
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
    for load in loads:
        if load.independent_rail_type == 'soft':
            for p in candidate_ics:
                if (p.name, load.name) in edges:
                    edge_to_load = edges[(p.name, load.name)]
                    for c in all_ic_and_load_nodes:
                        if c.name != load.name and (p.name, c.name) in edges:
                            model.AddImplication(edge_to_load, edges[(p.name, c.name)].Not())
    if constraints.get('max_sleep_current', 0) > 0:
        ic_sleep_terms = [int(ic.operating_current * SCALE) * ic_is_used[ic.name] for ic in candidate_ics]
        always_on_loads = [l for l in loads if l.always_on_in_sleep]
        load_sleep_total = int(sum(load.current_sleep for load in always_on_loads) * SCALE)
        model.Add(sum(ic_sleep_terms) + load_sleep_total <= int(constraints['max_sleep_current'] * SCALE))
    cost_objective = sum(int(ic.cost * 10000) * ic_is_used[ic.name] for ic in candidate_ics)
    model.Minimize(cost_objective)
    print("✅ 모델 생성 완료!")
    return model, edges, ic_is_used

def check_solution_validity(solution, candidate_ics, loads, battery, constraints):
    """주어진 해답이 모든 제약조건을 만족하는지 수동으로 검증하는 함수"""
    print("  -> 검증 중...", end="")
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    is_valid = True
    parent_to_children = defaultdict(list)
    for p, c in solution['active_edges']: parent_to_children[p].append(c)
    # 1. 전류 한계 검증
    for p_name, children_names in parent_to_children.items():
        if p_name not in candidate_ics_map: continue
        parent_ic = candidate_ics_map[p_name]
        actual_i_out = 0
        for c_name in children_names:
            if c_name in loads_map: actual_i_out += loads_map[c_name].current_active
            elif c_name in candidate_ics_map:
                child_ic = candidate_ics_map[c_name]
                child_children = parent_to_children.get(c_name, [])
                child_i_out = sum(loads_map[gc_name].current_active for gc_name in child_children if gc_name in loads_map)
                actual_i_out += child_ic.calculate_input_current(child_ic.vin, child_i_out)
        limit = parent_ic.i_limit * (1 + constraints.get('current_margin', 0.1))
        if actual_i_out > limit: is_valid = False; break
    if not is_valid: return False
    # 2. Independent Rail 검증
    independent_loads = {l.name for l in loads if l.independent_rail_type == 'soft'}
    for p_name, children_names in parent_to_children.items():
        children_set = set(children_names)
        if children_set.intersection(independent_loads) and len(children_set) > 1: is_valid = False; break
    if not is_valid: return False
    # 3. Power Sequence 검증
    for rule in constraints.get('power_sequences', []):
        j, k = rule['j'], rule['k']
        j_parent, k_parent = None, None
        for p, c in solution['active_edges']:
            if c == j: j_parent = p
            if c == k: k_parent = p
        if j_parent is not None and j_parent == k_parent: is_valid = False; break
    if not is_valid: return False
    print(" -> ✅ 유효")
    return True

# ================================================================
# 6. 리포팅 및 시각화 함수
# ================================================================
def visualize_tree(solution, candidate_ics, loads, battery, constraints, junction_temps, i_ins, i_outs, actual_i_ins_sleep, total_active_power, total_active_current, total_sleep_current):
    """솔루션 시각화 함수"""
    dot = Digraph(comment=f"Power Tree - Cost ${solution['cost']:.2f}", format='png')
    dot.attr('node', shape='box', style='rounded', fontname='Arial')
    margin_info = f"Current Margin: {constraints.get('current_margin', 0)*100:.0f}%"
    temp_info = f"Ambient Temp: {constraints.get('ambient_temperature', 25)}°C"
    dot.attr(rankdir='LR', label=f"{margin_info}\n{temp_info}\n\nSolution Cost: ${solution['cost']:.2f}", labelloc='t', fontname='Arial')
    battery_label = (
        f"🔋 {battery.name}\n\n"
        f"Total Active Power: {total_active_power:.2f} W\n"
        f"Total Active Current: {total_active_current * 1000:.1f} mA\n"
        f"Total Sleep Current: {total_sleep_current * 1000000:,.1f} µA"
    )
    dot.node(battery.name, battery_label, shape='Mdiamond', color='darkgreen')
    used_ics_map = {ic.name: ic for ic in candidate_ics if ic.name in solution['used_ic_names']}
    for ic_name, ic in used_ics_map.items():
        calculated_tj = junction_temps.get(ic_name, 0)
        i_in = i_ins.get(ic_name, 0); i_out = i_outs.get(ic_name, 0)
        i_in_sleep = actual_i_ins_sleep.get(ic_name, 0)
        thermal_margin = ic.t_junction_max - calculated_tj
        color = 'blue'
        if thermal_margin < 10: color = 'red'
        elif thermal_margin < 25: color = 'orange'
        label = (
            f"📦 {ic.name.split('@')[0]}\n\n"
            f"Vin: {ic.vin:.2f}V, Vout: {ic.vout:.2f}V\n"
            f"Iin: {i_in*1000:.1f}mA (Active) | {i_in_sleep*1000000:,.1f}µA (Sleep)\n"
            f"Iout: {i_out*1000:.1f}mA (Active)\n"
            f"Tj: {calculated_tj:.1f}°C (Max: {ic.t_junction_max}°C)\n"
            f"Iq: {ic.quiescent_current * 1000000:,.1f}µA\n"
            f"Cost: ${ic.cost:.2f}"
        )
        dot.node(ic_name, label, color=color, shape='box', penwidth='3')
    sequenced_loads = set()
    if 'power_sequences' in constraints:
        for seq in constraints['power_sequences']:
            sequenced_loads.add(seq['j']); sequenced_loads.add(seq['k'])
    for load in loads:
        label = f"💡 {load.name}\nActive: {load.voltage_typical}V | {load.current_active*1000:.1f}mA\n"
        if load.current_sleep > 0: label += f"Sleep: {load.current_sleep * 1000000:,.1f}µA\n"
        conditions = []
        if load.independent_rail_type: conditions.append("🔒 Independent")
        if load.name in sequenced_loads: conditions.append("⛓️ Sequence")
        if conditions: label += " ".join(conditions)
        penwidth = '1'
        if load.current_sleep > 0: penwidth = '3'
        dot.node(load.name, label, color='dimgray', penwidth=penwidth)
    for p_name, c_name in solution['active_edges']:
        dot.edge(p_name, c_name)
    print(f"\n🖼️  Generating diagram for solution with cost ${solution['cost']:.2f}...")
    return dot

def print_and_visualize_one_solution(solution, candidate_ics, loads, battery, constraints, solution_index=0):
    """주어진 단일 솔루션에 대해 상세 계산, 텍스트 출력, 시각화를 수행하는 함수"""
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    print(f"\n{'='*20} 솔루션 (비용: ${solution['cost']:.2f}) {'='*20}")
    used_ic_objects = [ic for ic in candidate_ics if ic.name in solution['used_ic_names']]
    actual_current_draw = {load.name: load.current_active for load in loads}
    sleep_current_draw = {load.name: load.current_sleep for load in loads}
    junction_temps, actual_i_ins, actual_i_outs, actual_i_ins_sleep = {}, {}, {}, {}
    processed_ics = set()
    while len(processed_ics) < len(used_ic_objects):
        for ic in used_ic_objects:
            if ic.name in processed_ics: continue
            children_names = [c for p, c in solution['active_edges'] if p == ic.name]
            if all(c_name in actual_current_draw for c_name in children_names):
                total_i_out = sum(actual_current_draw[c_name] for c_name in children_names); actual_i_outs[ic.name] = total_i_out
                i_in = ic.calculate_input_current(vin=ic.vin, i_out=total_i_out); actual_current_draw[ic.name] = i_in; actual_i_ins[ic.name] = i_in
                power_loss = ic.calculate_power_loss(vin=ic.vin, i_out=total_i_out); ambient_temp = constraints.get('ambient_temperature', 25)
                junction_temps[ic.name] = ambient_temp + (power_loss * ic.theta_ja)
                total_i_out_sleep = sum(sleep_current_draw.get(c_name, 0) for c_name in children_names if c_name in loads_map) # Use .get for safety
                ic_self_consumption = ic.quiescent_current
                if total_i_out_sleep > 0: ic_self_consumption = ic.operating_current
                i_in_sleep = 0
                if isinstance(ic, LDO): i_in_sleep = total_i_out_sleep + ic_self_consumption
                elif isinstance(ic, BuckConverter):
                    p_out_sleep = ic.vout * total_i_out_sleep; p_in_sleep = (p_out_sleep / 0.8) if 0.8 > 0 else 0
                    i_in_sleep = (p_in_sleep / ic.vin) + ic_self_consumption
                actual_i_ins_sleep[ic.name] = i_in_sleep; sleep_current_draw[ic.name] = i_in_sleep
                processed_ics.add(ic.name)
    primary_ics = [c_name for p_name, c_name in solution['active_edges'] if p_name == battery.name]
    total_active_current = sum(actual_i_ins.get(ic_name, 0) for ic_name in primary_ics)
    total_sleep_current = sum(actual_i_ins_sleep.get(ic_name, 0) for ic_name in primary_ics)
    battery_avg_voltage = (battery.voltage_min + battery.voltage_max) / 2
    total_active_power = battery_avg_voltage * total_active_current
    print(f"   - 시스템 전체 슬립 전류: {total_sleep_current * 1000:.4f} mA")
    print("\n--- Power Tree 구조 ---")
    tree_topology = defaultdict(list)
    for p, c in solution['active_edges']: tree_topology[p].append(c)
    def format_node_name(name, show_instance_num=False):
        if name in candidate_ics_map:
            ic = candidate_ics_map[name]; base_name = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
            if show_instance_num and '_copy' in ic.name: return f"{base_name} [#{ic.name.split('_copy')[-1]}]"
            return base_name
        elif name in loads_map: return f"💡 {name}"
        elif name == battery.name: return f"🔋 {name}"
        return name
    def print_instance_tree(parent_name, prefix=""):
        children = sorted(tree_topology.get(parent_name, []))
        for i, child_name in enumerate(children):
            is_last = (i == len(children) - 1); connector = "└── " if is_last else "├── "
            print(prefix + connector + format_node_name(child_name, show_instance_num=True))
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_instance_tree(child_name, new_prefix)
    print(format_node_name(battery.name))
    root_children = sorted(tree_topology.get(battery.name, []))
    for i, child_instance_name in enumerate(root_children):
        is_last = (i == len(root_children) - 1); connector = "└── " if is_last else "├── "
        print(connector + format_node_name(child_instance_name, show_instance_num=True))
        new_prefix = "    " if is_last else "│   "
        print_instance_tree(child_instance_name, new_prefix)
    
    # === VS Code 변경점: display() 대신 파일로 저장하고 열기 ===
    dot_graph = visualize_tree(
        solution, candidate_ics, loads, battery, constraints,
        junction_temps, actual_i_ins, actual_i_outs, actual_i_ins_sleep,
        total_active_power, total_active_current, total_sleep_current
    )
    # 파일 이름에 솔루션 번호와 비용을 포함하여 중복 방지
    output_filename = f'solution_{solution_index}_cost_{solution["cost"]:.2f}'
    # .render()는 다이어그램을 파일로 저장하고, view=True는 해당 파일을 자동으로 열어줍니다.
    dot_graph.render(output_filename, view=True, cleanup=True, format='png')
    print(f"✅ 다이어그램을 '{output_filename}.png' 파일로 저장하고 실행했습니다.")
    
def find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints):
    """대표 해답을 분석하여, 동일 IC 그룹 내에서 Load를 재분배하는 모든 유효한 조합을 찾는 함수"""
    print("\n\n👑 --- 최종 단계: 모든 부하 분배 조합 탐색 --- 👑")
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    ic_type_to_instances = defaultdict(list)
    for ic_name in base_solution['used_ic_names']:
        ic = candidate_ics_map[ic_name]
        ic_type = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
        ic_type_to_instances[ic_type].append(ic)
    instance_to_children = defaultdict(set)
    for p, c in base_solution['active_edges']:
        if p in candidate_ics_map: instance_to_children[p].add(c)
    target_group = None
    for ic_type, instances in ic_type_to_instances.items():
        if len(instances) > 1:
            total_load_pool = set()
            for inst in instances: total_load_pool.update(instance_to_children[inst.name])
            target_group = {'instances': [inst.name for inst in instances], 'load_pool': list(total_load_pool)}
            break
    if not target_group:
        print("\n -> 이 해답에는 생성할 병렬해가 없습니다.")
        if check_solution_validity(base_solution, candidate_ics, loads, battery, constraints):
             print_and_visualize_one_solution(base_solution, candidate_ics, loads, battery, constraints, solution_index=0)
        return
    def find_partitions(items, num_bins):
        if not items: yield [[] for _ in range(num_bins)]
        else:
            for partition in find_partitions(items[1:], num_bins):
                for i in range(num_bins): yield partition[:i] + [[items[0]] + partition[i]] + partition[i+1:]
                if num_bins > len(partition): yield partition + [[items[0]]]
    valid_solutions, seen_partitions = [], set()
    num_instances, load_pool = len(target_group['instances']), target_group['load_pool']
    for p in find_partitions(load_pool, num_instances):
        if len(p) == num_instances:
            canonical_partition = tuple(sorted([tuple(sorted(sublist)) for sublist in p]))
            if canonical_partition in seen_partitions: continue
            seen_partitions.add(canonical_partition)
            new_edges = [edge for edge in base_solution['active_edges'] if edge[0] not in target_group['instances']]
            for i, instance_name in enumerate(target_group['instances']):
                for load_name in p[i]: new_edges.append((instance_name, load_name))
            new_solution = {"used_ic_names": base_solution['used_ic_names'], "active_edges": new_edges, "cost": base_solution['cost']}
            if check_solution_validity(new_solution, candidate_ics, loads, battery, constraints):
                valid_solutions.append(new_solution)
    print(f"\n✅ 총 {len(valid_solutions)}개의 유효한 병렬해 구조를 찾았습니다.")
    for i, solution in enumerate(valid_solutions):
        print(f"\n--- [병렬해 #{i+1}] ---")
        print_and_visualize_one_solution(solution, candidate_ics, loads, battery, constraints, solution_index=i+1)

# ================================================================
# 7. 메인 실행 블록
# ================================================================
def main():
    """메인 실행 함수"""
    # 1. 설정 로드
    battery, available_ics, loads, constraints = load_configuration(json_config_string)
     
    # 2. 후보 IC 생성
    candidate_ics, ic_groups = expand_ic_instances(available_ics, loads, battery, constraints)
    
    # 3. CP-SAT 모델 생성
    model, edges, ic_is_used = create_solver_model(candidate_ics, loads, battery, constraints, ic_groups)
    
    # 4. 솔버 생성 및 대표 해 찾기
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solution_logger = SolutionLogger(ic_is_used, edges, limit=1) # 가장 좋은 해 1개만 찾음
    
    print("\n🔍 최적의 대표 솔루션 탐색 시작...")
    status = solver.Solve(model, solution_logger)
    
    # 5. 결과 처리
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and solution_logger.solutions:
        print(f"\n🎉 탐색 완료! (상태: {solver.StatusName(status)})")
        # 찾은 대표 해를 기반으로 모든 가능한 부하 분배 조합을 탐색
        base_solution = solution_logger.solutions[0]
        base_solution['cost'] = base_solution['score'] / 10000 # score를 실제 cost로 변환
        find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints)
    else:
        print("\n❌ 유효한 솔루션을 찾지 못했습니다.")

if __name__ == "__main__":
    main()