# main.py

import json
from ortools.sat.python import cp_model

# SolutionLogger는 더 이상 사용하지 않으므로 임포트에서 제외
from pocat_core import (
    load_configuration, expand_ic_instances, create_solver_model,
    find_all_load_distributions
)
from pocat_visualizer import (
    check_solution_validity, print_and_visualize_one_solution
)

def main():
    """메인 실행 함수"""
    # 1. 설정 로드
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            json_config_string = f.read()
    except FileNotFoundError:
        print("오류: 설정 파일 'config.json'을(를) 찾을 수 없습니다.")
        return
        
    battery, available_ics, loads, constraints = load_configuration(json_config_string)
    
    # 2. 후보 IC 생성
    candidate_ics, ic_groups = expand_ic_instances(available_ics, loads, battery, constraints)
    
    # 3. CP-SAT 모델 생성
    model, edges, ic_is_used = create_solver_model(candidate_ics, loads, battery, constraints, ic_groups)
    
    # 4. 솔버 생성 및 탐색 시간 설정
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 300.0 # 최대 30초간 최적해 탐색
    
    # 5. 솔버 실행 (SolutionLogger 없이)
    print("\n🔍 최적의 대표 솔루션 탐색 시작...")
    status = solver.Solve(model)
    
    # 6. 결과 처리
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n🎉 탐색 완료! (상태: {solver.StatusName(status)})")
        
        # 탐색이 끝난 solver에서 직접 결과값을 가져와 base_solution 구성
        base_solution = {
            "score": solver.ObjectiveValue(),
            "cost": solver.ObjectiveValue() / 10000,
            "used_ic_names": {name for name, var in ic_is_used.items() if solver.Value(var)},
            "active_edges": [(p, c) for (p, c), var in edges.items() if solver.Value(var)]
        }
        
        # 대표해를 기반으로 병렬해 탐색
        all_solutions = find_all_load_distributions(
            base_solution, candidate_ics, loads, battery, constraints,
            viz_func=print_and_visualize_one_solution,
            check_func=check_solution_validity
        )
        
    else:
        print("\n❌ 유효한 솔루션을 찾지 못했습니다.")

if __name__ == "__main__":
    main()