from ortools.sat.python import cp_model

def SimpleSatProgram():
    """가장 기본적인 CP-SAT 예제."""
    # 모델 생성
    model = cp_model.CpModel()

    # 변수 생성
    # x는 0, 1, 2 중 하나의 값을 가짐
    x = model.NewIntVar(0, 2, 'x')
    # y는 0, 1 중 하나의 값을 가짐
    y = model.NewIntVar(0, 1, 'y')

    # 제약 조건 추가
    model.Add(x != y)

    # 솔버 생성 및 해결
    solver = cp_model.CpSolver()
    print("간단한 모델로 솔버를 실행합니다...")
    status = solver.Solve(model)
    print("실행 완료!")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'x = {solver.Value(x)}')
        print(f'y = {solver.Value(y)}')
    else:
        print('해결책을 찾지 못했습니다.')

SimpleSatProgram()