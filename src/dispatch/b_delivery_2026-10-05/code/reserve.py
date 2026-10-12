"""Provisional, frozen uncertainty-to-energy-buffer mapping; not a reliability claim."""
import math


def reserve_profile(window, config):
    """Half interval width maps to capped kW, then a duration maps to kWh.

    Reserve is a combined deliverable-energy stock buffer, not a promise that
    the entire power can be delivered instantly under startup/ramp constraints.
    """
    lo, hi = config['min_kw'], config['max_kw']
    scale, duration = config['width_scale'], config['duration_h']
    if not (0 <= lo <= hi and scale >= 0 and duration > 0):
        raise ValueError('Invalid reserve mapping configuration')
    answer = []
    for item in window:
        values = [item[k] for k in ('q10_kw', 'q50_kw', 'q90_kw')]
        if not all(math.isfinite(v) for v in values) or not values[0] <= values[1] <= values[2]:
            raise ValueError('Invalid quantiles for reserve')
        answer.append(min(hi, max(lo, scale * (values[2] - values[0]))) * duration)
    return answer


def available_buffer_kwh(state, p):
    b, h, fc = p['battery'], p['hydrogen_tank'], p['fuel_cell']
    return max(0.0, (state.battery_energy_kwh - b['soc_min'] * b['energy_capacity_kwh'])
               * b['discharge_efficiency']) + max(0.0, state.hydrogen_kg - h['min_kg']) * fc['electric_output_kwh_per_kg_h2']
