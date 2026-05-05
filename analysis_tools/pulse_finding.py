import numpy as np


def do_pulse_finding(waveform, debug=False):
    # mimics the pulse finding on the mPMT, single sample at a time

    threshold = 20
    fIntegralPreceding = 4;
    fIntegralFollowing = 2;

    above_threshold = np.where(waveform[3:-2] > threshold)[0] + 3  # any samples over the threshold from 3 -> -2
    if debug:
        print("above theshold", np.sum(above_threshold))
    local_max_condition = np.full(len(above_threshold), False)
    integral_condition = np.full(len(above_threshold), False)
    sufficient_period = np.full(len(above_threshold), False)
    pulses_found = []
    last_index = 0
    for i, index in enumerate(above_threshold):
        if debug:
            print("On max", i, index)
        max_point = waveform[index]

        if debug:
            print("max waveform", max_point)
            print("waveform[index-1]", waveform[index - 1])
            print("waveform[index+1]", waveform[index + 1])
            print("waveform[index-2]", waveform[index - 2])
            print("waveform[index+2]", waveform[index + 2])

        if waveform[index] <= waveform[index - 1]: continue;
        if waveform[index] < waveform[index + 1]: continue;
        if waveform[index] <= waveform[index + 2]: continue;
        if waveform[index] <= waveform[index - 2]: continue;

        local_max_condition[i] = True

        # integral
        start = index - fIntegralPreceding if (index - fIntegralPreceding) > 0 else 0
        end = index + fIntegralFollowing + 1 if (index + fIntegralFollowing + 1) < len(waveform) else len(waveform)
        integral = np.sum(waveform[start:end])
        if debug:
            print("Integral from", start, end, " is ", integral, waveform[start:end])
            # print("Integral from",start,end-1," is ", np.sum(waveform[start:end-1]))

        if integral < threshold * 2:
            continue
        integral_condition[i] = True

        # sufficient period
        if (last_index > 0) and (index - last_index) <= 20:
            continue
        sufficient_period[i] = True

        pulses_found.append(index)
        last_index = index
        if debug:
            print("Pulse found amp", waveform[index], "integral", integral)
    # return pulses_found, above_threshold, local_max_condition, integral_condition, sufficient_period
    return pulses_found


def do_pulse_finding_vect(wf, debug=False):
    # a more vectorised faster approach to the above function doing the same thing but on a vector of waveforms to process all weveforms
    # simultaneously

    # 1. Thresholding
    threshold = 20
    fIntegralPreceding = 4;
    fIntegralFollowing = 2;

    amp_mask = wf > threshold

    # 2. Local maxima condition (centered at idx)
    local_max = np.full_like(wf, False, dtype=bool)
    # Ensure enough room for lookaround
    valid = np.full_like(wf, True, dtype=bool)
    valid[:, :2] = False
    valid[:, -2:] = False

    # Conditions
    c1 = wf > np.roll(wf, 1, axis=1)  # wf[i] > wf[i-1]
    c2 = wf >= np.roll(wf, -1, axis=1)  # wf[i] >= wf[i+1]
    c3 = wf > np.roll(wf, 2, axis=1)  # wf[i] > wf[i-2]
    c4 = wf > np.roll(wf, -2, axis=1)  # wf[i] > wf[i+2]

    local_max = amp_mask & valid & c1 & c2 & c3 & c4

    if (debug):
        print(np.sum(local_max), np.sum(amp_mask), np.sum(c1), np.sum(c2), np.sum(c3), np.sum(c4))

    # 3. Integral calculation (sum over [i-4:i+3], 7 bins)
    integral = np.zeros_like(wf, dtype=float)
    for i in range(-fIntegralPreceding, fIntegralFollowing + 1):
        integral += np.roll(wf, -i, axis=1)
        if (debug):
            print("int", i, np.roll(wf, i, axis=1)[0][8], integral[0][8])
    # Mask invalid edges in integral (avoid wraparound)
    integral[:, :fIntegralPreceding] = 0
    integral[:, -fIntegralFollowing:] = 0

    # 4. Apply integral condition
    integral_mask = integral > threshold * 2

    # 5. Final pulse candidate mask
    pulse_mask = local_max & integral_mask

    all_indices = []
    for row in pulse_mask:
        peak_indices = np.where(row)[0]
        last_index = -20
        pulses = []
        for index in peak_indices:
            if index - last_index > 20:
                pulses.append(index)
                last_index = index
        all_indices.append(pulses)

    return all_indices
        