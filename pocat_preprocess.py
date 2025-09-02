# =========================
# Dominance pruning & grouping (전처리)
# =========================
from collections import defaultdict
from typing import List, Dict, Tuple

def _norm_type(t: str) -> str:
    """타입 명칭 정규화: Buck -> DCDC 로 통일"""
    if not isinstance(t, str):
        return str(t)
    t = t.strip().upper()
    if t in ("BUCK", "DCDC", "DC-DC", "SWITCHER"):
        return "DCDC"
    if t in ("LDO", "LINEAR"):
        return "LDO"
    return t  # PMIC 등 그 외 그대로

def _get(ic: Dict, key: str, default=None):
    """dict 안전 접근 + None 방어"""
    v = ic.get(key, default)
    return default if v is None else v

def _extract_specs(ic: Dict) -> Dict:
    """
    OR-Tools 모델링에 'Feasibility' 관여 축만 추출.
    비용/면적/손실 등 목적함수 축은 제외 (전처리에서 건드리지 않기 위함).
    """
    typ = _norm_type(_get(ic, "type", "DCDC"))
    vin_min = float(_get(ic, "vin_min", _get(ic, "V_in_min", 0.0)))
    vin_max = float(_get(ic, "vin_max", _get(ic, "V_in_max", 999.0)))

    # vout: range 또는 fixed를 모두 지원
    if "vout_fixed" in ic:
        vout_min = vout_max = float(ic["vout_fixed"])
    elif "V_out" in ic and isinstance(ic["V_out"], (int, float)):
        vout_min = vout_max = float(ic["V_out"])
    else:
        vout_min = float(_get(ic, "vout_min", _get(ic, "V_out_min", 0.0)))
        vout_max = float(_get(ic, "vout_max", _get(ic, "V_out_max", 999.0)))

    # 출력 전류 (용량)
    i_limit = float(_get(ic, "i_limit", _get(ic, "I_out_max", 0.0)))

    # LDO dropout (작을수록 우위)
    v_dropout = float(_get(ic, "v_dropout_min", _get(ic, "V_dropout_min", 999.0)))

    # 온도 스펙(선택적으로 포함)
    theta_ja = float(_get(ic, "theta_ja", _get(ic, "RθJA", 0.0)))
    tj_max   = float(_get(ic, "t_junction_max", _get(ic, "Tj_max", 0.0)))

    cost = float(_get(ic, "cost", 1e9))  # 없으면 매우 큰 비용으로 가정

    return dict(
        type=typ, vin_min=vin_min, vin_max=vin_max,
        vout_min=vout_min, vout_max=vout_max,
        i_limit=i_limit, v_dropout=v_dropout,
        theta_ja=theta_ja, tj_max=tj_max, cost=cost
    )

def _dominates_b_over_a(a_spec: Dict, b_spec: Dict) -> bool:
    """
    'Feasibility' 지배만 판단:
      - type 동일
      - Vin 범위: b가 a 포함
      - Vout 범위: b가 a 포함 (고정값/범위 모두)
      - I_limit: b >= a
      - (LDO) Dropout: b <= a
      - (선택) 온도: b.tj_max >= a.tj_max  (너가 온도 제약 쓰면 이 축 유지)
      - 비용: b.cost <= a.cost
      - 그리고 '최소 하나'는 엄격 우위
    목적함수(손실/면적/Iq 등)는 여기서 건드리지 않는다 → 안전 전처리.
    """
    if a_spec["type"] != b_spec["type"]:
        return False

    include_vin  = (b_spec["vin_min"] <= a_spec["vin_min"] and b_spec["vin_max"] >= a_spec["vin_max"])
    include_vout = (b_spec["vout_min"] <= a_spec["vout_min"] and b_spec["vout_max"] >= a_spec["vout_max"])
    i_ok = (b_spec["i_limit"] >= a_spec["i_limit"])
    ld_ok = True
    if a_spec["type"] == "LDO":
        ld_ok = (b_spec["v_dropout"] <= a_spec["v_dropout"])
    tj_ok = (b_spec["tj_max"] >= a_spec["tj_max"])
    cost_ok = (b_spec["cost"] <= a_spec["cost"])

    if not (include_vin and include_vout and i_ok and ld_ok and tj_ok and cost_ok):
        return False

    strict = (
        (b_spec["vin_min"] < a_spec["vin_min"]) or
        (b_spec["vin_max"] > a_spec["vin_max"]) or
        (b_spec["vout_min"] < a_spec["vout_min"]) or
        (b_spec["vout_max"] > a_spec["vout_max"]) or
        (b_spec["i_limit"]  > a_spec["i_limit"])  or
        (a_spec["type"] == "LDO" and b_spec["v_dropout"] < a_spec["v_dropout"]) or
        (b_spec["tj_max"]   > a_spec["tj_max"])   or
        (b_spec["cost"]     < a_spec["cost"])
    )
    return strict

def prune_dominated_ic_instances(ic_list: List[Dict]) -> Tuple[List[Dict], List[Tuple[int,int]]]:
    """
    입력: IC dict 리스트 (확장/복제 포함)
    출력:
      - 지배 제거 후 남긴 리스트(new_ics)
      - (선택적) 경쟁자 컷용 지배쌍 리스트 dominated_pairs: (a_idx, b_idx) 의미: b가 a를 지배(b dominates a)
    """
    specs = [_extract_specs(ic) for ic in ic_list]
    keep = [True]*len(ic_list)
    dominated_pairs: List[Tuple[int,int]] = []

    # O(n^2) 비교 (n=1,188 정도는 충분히 빠름)
    for i, a in enumerate(specs):
        if not keep[i]:
            continue
        for j, b in enumerate(specs):
            if i == j or not keep[i]:
                continue
            if _dominates_b_over_a(a, b):
                # j(b)가 i(a)를 지배 → a는 버려도 안전
                keep[i] = False
                dominated_pairs.append((i, j))
                break

    new_ics = []
    old_to_new = {}
    for old_idx, (ic, k) in enumerate(zip(ic_list, keep)):
        if k:
            old_to_new[old_idx] = len(new_ics)
            # id/uid 보정: 없으면 새로 부여
            ic = dict(ic)
            if "id" not in ic and "uid" not in ic:
                ic["uid"] = f"IC_{len(new_ics)}"
            new_ics.append(ic)

    # 지배쌍 인덱스도 새 인덱스로 재매핑 (남아있는 것만 유지)
    remapped_pairs = []
    for a_old, b_old in dominated_pairs:
        if keep[a_old] or keep[b_old]:
            # a_old는 버려졌으니 보통 안 남음. 혹시 둘 다 남아있으면(동률 케이스) 컷으로 쓰도록 남김
            if a_old in old_to_new and b_old in old_to_new:
                remapped_pairs.append((old_to_new[a_old], old_to_new[b_old]))

    return new_ics, remapped_pairs

def group_competitor_families(ic_list: List[Dict]) -> List[List[int]]:
    """
    '동일/유사 스펙 복제본' 묶음 생성.
    - 같은 type, 같은 (vin_min/max, vout_min/max), 같은 i_limit, (LDO면 dropout) 기준으로 그룹화
    - 그룹 내에서 cost 오름차순 정렬 → prefix 제약 적용하기 좋음
    """
    buckets = defaultdict(list)

    def key_of(ic: Dict):
        s = _extract_specs(ic)
        # 부동소수 노이즈 줄이려고 round
        return (
            s["type"],
            round(s["vin_min"], 4), round(s["vin_max"], 4),
            round(s["vout_min"], 4), round(s["vout_max"], 4),
            round(s["i_limit"], 4),
            round(s["v_dropout"], 4) if s["type"] == "LDO" else None,
        )

    for idx, ic in enumerate(ic_list):
        buckets[key_of(ic)].append(idx)

    families: List[List[int]] = []
    for _, idxs in buckets.items():
        if len(idxs) <= 1:
            continue
        # 비용 오름차순 정렬
        idxs.sort(key=lambda i: _extract_specs(ic_list[i])["cost"])
        families.append(idxs)
    return families
