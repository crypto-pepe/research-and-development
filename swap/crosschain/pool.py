from dataclasses import dataclass, field
from functools import reduce
from copy import deepcopy

MAX_DIFF = -0.00001844674

@dataclass
class PoolState:
    weights: list[int]
    max_allocation: int
    weight_amplitude: int
    slippage_rate: int
    fee_max: int
    fee_base: int
    assets_balances: list[int] = field(default_factory=lambda: [])
    assets_total_supply: int = 0
    total_supply: int = 0
    
def create(weights: list[int], max_allocation: int, weight_amplitude: int, slippage_rate: int, fee_max: int, fee_base: int):
    [validate_weight(weight) for weight in weights]
    validate_fee_max(fee_max)
    validate_max_allocation(max_allocation)
    validate_weight_amplitude(max_allocation, weight_amplitude)
    validate_slippage_rate(slippage_rate)
    validate_fee_base(fee_base)
    
    return PoolState(weights, max_allocation, weight_amplitude, slippage_rate, fee_max, fee_base)

def proportional_deposit(amount, state: PoolState):
    prev_state = deepcopy(state)
    
    if state.assets_total_supply == 0:
        state.assets_balances = list(map(lambda weight: weight * amount , state.weights))
    else:
        ratio = amount / state.assets_total_supply
        state.assets_balances = list(map(lambda balance: balance * (1 + ratio) , state.assets_balances))
        
    mint_amount = amount if state.total_supply == 0 else amount / state.assets_total_supply * state.total_supply
    state.total_supply += mint_amount
    state.assets_total_supply = sum(state.assets_balances)
    validate_invariant(prev_state, state)
    
    return (mint_amount, list(map(lambda i: i[1]-i[0], zip(prev_state.assets_balances, state.assets_balances))))

def deposit(amounts: [int], state: PoolState):
    prev_state = deepcopy(state)
    if not state.assets_balances:
        state.assets_balances = list(map(lambda i: 0, amounts))
        
    state.assets_balances = list(map(lambda i: i[0]+i[1], zip(amounts, state.assets_balances)))
    state.assets_total_supply = sum(state.assets_balances)
    validate_allocation(prev_state, state)
    
    prev_fee = calculate_fee(prev_state)
    fee = calculate_fee(state)
    if prev_state.total_supply == 0:
        state.total_supply = state.assets_total_supply - fee
        return state.total_supply
    else:
        assets_diff = state.assets_total_supply - prev_state.assets_total_supply
        fee_diff = fee - prev_fee
        rate = assets_diff - fee_diff
        rate /= (prev_state.assets_total_supply - prev_fee)
        state.total_supply += prev_state.total_supply * rate
        validate_invariant(prev_state, state)
        return prev_state.total_supply * rate

def proportional_withdraw(amount: int, state: PoolState):
    prev_state = deepcopy(state)
    rate = amount * (1 - state.fee_base) / state.total_supply
    
    state.total_supply -= amount
    state.assets_balances = list(map(lambda balance: balance * (1 - rate), state.assets_balances))
    state.assets_total_supply = sum(state.assets_balances)
    validate_invariant(prev_state, state)
    
    return list(map(lambda i: i[0]-i[1], zip(prev_state.assets_balances, state.assets_balances)))

def withdraw(amounts: [int], state: PoolState):
    prev_state = deepcopy(state)
    state.assets_balances = list(map(lambda i: i[1]-i[0], zip(amounts, state.assets_balances)))
    state.assets_total_supply = sum(state.assets_balances)
    
    validate_allocation(prev_state, state)
    
    prev_fee = calculate_fee(prev_state)
    fee = calculate_fee(state)
    
    if prev_state.total_supply == 0:
        state.total_supply = state.assets_total_supply - fee
        return state.total_supply * (1 + state.fee_base)
    else:
        assets_diff = state.assets_total_supply - prev_state.assets_total_supply
        fee_diff = fee - prev_fee
        rate = assets_diff - fee_diff
        rate /= (prev_state.assets_total_supply - prev_fee)
        state.total_supply += prev_state.total_supply * rate
        validate_invariant(prev_state, state)
        return prev_state.total_supply * rate * (1 + state.fee_base)

def withdraw_bisection(target: int, lpAmount: int, state: PoolState):
    prev_state = deepcopy(state)
    prev_fee = calculate_fee(prev_state)
    prev_util = prev_state.assets_total_supply - prev_fee
    k = prev_state.total_supply / (prev_state.total_supply - lpAmount)
    ref_util_value = prev_util / k

    start = 0
    stop = lpAmount + prev_fee

    for i in range(64):
        current = (stop + start) / 2
        temp_state = deepcopy(prev_state)
        temp_state.assets_balances[target] -= current
        temp_state.assets_total_supply -= current
        util_value = temp_state.assets_total_supply - calculate_fee(temp_state)

        if util_value - ref_util_value < 0.0001 and util_value - ref_util_value >= 0:
            state.assets_total_supply = prev_state.assets_total_supply - current
            state.assets_balances[target] = prev_state.assets_balances[target] - current
            state.total_supply -= lpAmount
        
            validate_allocation(prev_state, state)
            validate_invariant(prev_state, state)

            return current * (1 - state.fee_base), i
        else:
            if util_value - ref_util_value < 0:
                stop = current
            else:
                start = current

    assert False, f"can't evaluate target amount"

def swap_bisection(source: int, target: int, amount: int, state: PoolState):
    assert (source != target), f"expected source != target"
    prev_state = deepcopy(state)
    prev_fee = calculate_fee(prev_state)

    current_state = deepcopy(prev_state)
    current_state.assets_balances[source] += amount
    current_state.assets_total_supply += amount

    start = 0
    stop = amount + prev_fee
    ref_util_value = prev_state.assets_total_supply - prev_fee

    for i in range(64):
        current = (stop + start) / 2
        temp_state = deepcopy(current_state)
        temp_state.assets_balances[target] -= current
        temp_state.assets_total_supply -= current
        util_value = temp_state.assets_total_supply - calculate_fee(temp_state)

        if util_value - ref_util_value < 0.0001 and util_value - ref_util_value >= 0:
            state.assets_total_supply = prev_state.assets_total_supply + amount - current
            state.assets_balances[target] = prev_state.assets_balances[target] - current
            state.assets_balances[source] = prev_state.assets_balances[source] + amount

            validate_allocation(prev_state, state)
            validate_swap_invariant(prev_state, state)

            return current * (1 - state.fee_base), i
        else:
            if util_value - ref_util_value < 0:
                stop = current
            else:
                start = current

    assert False, f"can't evaluate target amount"

def swap_iterations(source: int, target: int, amount: int, state: PoolState):
    assert (source != target), f"expected source != target"
    prev_state = deepcopy(state)
    prev_fee = calculate_fee(prev_state)

    current_state = deepcopy(prev_state)
    current_state.assets_balances[source] += amount
    current_state.assets_total_supply += amount

    target_amount = amount
    for i in range(25):
        temp_state = deepcopy(current_state)
        temp_state.assets_balances[target] -= target_amount
        temp_state.assets_total_supply -= target_amount
        fee = calculate_fee(temp_state)
        fee_diff = fee - prev_fee
        temp_amount = target_amount - fee_diff

        assert temp_amount > 0, "temp amount less than 0"
        if abs(target_amount - temp_amount) < 0.01:
            target_amount = temp_amount
            state.assets_total_supply = prev_state.assets_total_supply + amount - target_amount
            state.assets_balances[target] = prev_state.assets_balances[target] - target_amount
            state.assets_balances[source] = prev_state.assets_balances[source] + amount
            validate_allocation(prev_state, state)
            validate_swap_invariant(prev_state, state)
            return target_amount * (1 - state.fee_base)
        else:
            target_amount = temp_amount
    
    assert False, f"can't evaluate target amount"

def calculate_microfee(balance: int, equilibrium: int, state: PoolState):
    if balance < equilibrium:
        threshold = equilibrium * (1 - state.weight_amplitude)
        if balance < threshold:
            fee_margin = threshold - balance
            fee_rate = fee_margin / equilibrium * state.slippage_rate
            return fee_margin * state.fee_max if fee_rate > state.fee_max else fee_rate * fee_margin
    else:
        threshold = equilibrium * (1 + state.weight_amplitude)
        if balance > threshold:
            fee_margin = balance - threshold
            fee_rate = fee_margin / equilibrium * state.slippage_rate
            return fee_margin * state.fee_max if fee_rate > state.fee_max else fee_rate * fee_margin

    return 0

def calculate_fee(state: PoolState):
    return reduce(lambda acc, it: acc + calculate_microfee(it[0], state.assets_total_supply * it[1], state), 
                  zip(state.assets_balances, state.weights), 0)

def calculate_util_func(state: PoolState):
    return [state.assets_total_supply - calculate_fee(state), state.total_supply]

# validators
def validate_weight(weight: int):
    assert (0 < weight <= 1), f"expected 0 < weight <= 1, but weight={weight}"

def validate_max_allocation(max_allocation: int):
    assert (0 < max_allocation < 1), f"expected 0 < max_allocation < 1, but max_allocation={max_allocation}"

def validate_weight_amplitude(max_allocation: int, weight_amplitude: int):
    assert (0 < weight_amplitude < max_allocation), f"expected 0 < weight_amplitude < max_allocation, but weight_amplitude={weight_amplitude}"
    
def validate_slippage_rate(slippage_rate: int):
    assert (0 < slippage_rate), f"expected 0 < slippage_rate, but slippage_rate={slippage_rate}"
    
def validate_fee_max(fee_max: int):
    assert (fee_max > 0), f"expected fee_max > 0, but fee_max={fee_max}"
    
def validate_fee_base(fee_base: int):
    assert (1 > fee_base >= 0), f"expected 1 > fee_base >= 0, but fee_base={fee_base}"
    
def validate_invariant(prev_state: PoolState, state: PoolState):
    if prev_state.total_supply == 0 or state.total_supply == 0:
        return
    
    prev_assets_rate = (prev_state.assets_total_supply - calculate_fee(prev_state)) / prev_state.total_supply
    assets_rate = (state.assets_total_supply - calculate_fee(state)) / state.total_supply
    diff = assets_rate - prev_assets_rate
    assert (diff > 0 or diff >= MAX_DIFF), f"expected diff > 0 or diff >= MAX_DIFF, but diff={diff}"
    
def validate_swap_invariant(prev_state: PoolState, state: PoolState):
    util = state.assets_total_supply - calculate_fee(state)
    prev_util = prev_state.assets_total_supply - calculate_fee(prev_state)
    diff = util - prev_util 
    assert (diff > 0 or diff >= MAX_DIFF), f"expected diff > 0 or diff >= MAX_DIFF, but diff={diff}"
    
def validate_allocation(prev_state: PoolState, state: PoolState):
    equilibriums = list(map(lambda weight: weight * state.assets_total_supply, state.weights))
    [
        validate_allocation_for_asset(amount, prev_amount, equilibrium, weight, prev_state, state)
        for (amount, prev_amount, weight, equilibrium) in zip(state.assets_balances, prev_state.assets_balances, state.weights, equilibriums)
    ]

def validate_allocation_for_asset(amount: int, prev_amount: int, equilibrium: int, weight: int, prev_state: PoolState, state: PoolState):
    max_allocation = 1 + state.max_allocation if amount > equilibrium else 1 - state.max_allocation
    limit = equilibrium * max_allocation
    prev_limit = prev_state.assets_total_supply * weight * max_allocation
    
    if amount > equilibrium:
        if amount > limit:
            assert (prev_amount >= prev_limit), "similar direction of overweight is expected"
            assert (amount - limit < prev_amount - prev_limit), "overweight is expected to improve"
    else:
        if amount < limit:
            assert (prev_amount <= prev_limit), "similar direction of overweight is expected"
            assert (limit - amount >= prev_limit - prev_amount), "overweight is expected to improve"
