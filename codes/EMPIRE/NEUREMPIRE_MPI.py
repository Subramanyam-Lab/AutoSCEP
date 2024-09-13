def compute_local_sum_season(args):
    instance, i, w, s = args
    discount = value(instance.discount_multiplier[i])
    shed_cost = sum(
        value(instance.seasScale[s]) * value(instance.nodeLostLoadCost[n, i]) * value(instance.loadShed[n, h, i, w])
        for n in instance.Node
        for (s_inner, h) in instance.HoursOfSeason
        if s_inner == s
    )
    operational_cost = sum(
        value(instance.seasScale[s]) * value(instance.genMargCost[g, i]) * value(instance.genOperational[n, g, h, i, w])
        for (n, g) in instance.GeneratorsOfNode
        for (s_inner, h) in instance.HoursOfSeason
        if s_inner == s
    )
    return discount * (shed_cost + operational_cost)

def compute_expected_second_stage_value_parallel(instance, num_scenarios, seed):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 모든 시나리오를 리스트로 변환
    scenarios = list(instance.Scenario)
    total_scenarios = len(scenarios)

    # 시나리오를 프로세스 수에 따라 분할
    scenarios_per_proc = total_scenarios // size
    remainder = total_scenarios % size

    if rank < remainder:
        start = rank * (scenarios_per_proc + 1)
        end = start + scenarios_per_proc + 1
    else:
        start = rank * scenarios_per_proc + remainder
        end = start + scenarios_per_proc

    local_scenarios = scenarios[start:end]

    # 멀티프로세싱 풀 생성 (예: 4개의 스레드)
    pool = multiprocessing.Pool(processes=4)  # 원하는 프로세스 수로 조정
    tasks = []
    for w in local_scenarios:
        for s in instance.Season:
            for i in instance.PeriodActive:
                tasks.append((instance, i, w, s))

    # 병렬로 부분 합 계산 (시즌 단위)
    results = pool.map(compute_local_sum_season, tasks)
    local_sum = sum(results)
    pool.close()
    pool.join()

    # 모든 프로세스의 부분 합을 합산하여 최종 기대값 계산
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        expected_value = total_sum / num_scenarios
        return expected_value
    else:
        return None
