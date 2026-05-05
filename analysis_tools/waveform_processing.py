import numpy as np


class WaveformProcessingTeststand:
    # copied from process_parquet test stand methods for waveform processing
    def __init__(self):
        self.cfd_raw_t = [0.16323452713658082,
                          0.20385733509493395,
                          0.24339187740767365,
                          0.2822514122310461,
                          0.3208335490313887,
                          0.35953379168152044,
                          0.3987592183841288,
                          0.4389432980060811,
                          0.4805630068163285,
                          0.5241597383052767,
                          0.5703660640730557,
                          0.6199413381955754,
                          0.6738206794685682,
                          0.7331844507933303,
                          0.7995598000823612,
                          0.874973724581176,
                          0.9621917102137131,
                          1.0301530251726216,
                          1.0769047405430523,
                          1.1210801763323819,
                          1.1632345271365807]

        # correction for the amplitude derived by summing the largest three adcs
        self.amp_raw_t = [2.0413475167493225,
                          2.0642014124776784,
                          2.0847238089021274,
                          2.1028869067818117,
                          2.118667914530039,
                          2.1320484585033723,
                          2.1430140317025583,
                          2.151553497195665,
                          2.1576586607668613,
                          2.1613239251470255,
                          2.162546035746829,
                          2.1613239251470255,
                          2.1576586607668617,
                          2.1515534971956654,
                          2.143014031702558,
                          2.1320484585033723,
                          2.118667914530039,
                          2.1028869067818117,
                          2.0847238089021274,
                          2.0642014124776784,
                          2.0413475167493225]

        self.cfd_true_t = [-0.5 + 0.05 * i for i in range(21)]

    def get_peak_timebins(self, waveform, threshold):
        # the following is from process_root
        # note that the waveforms in the parquet file are already baseline subtracted and inverted
        # So reverse the sign of baseline and adcs in the get_cfd function
        # use the most frequent waveform value as the baseline
        values, counts = np.unique(waveform, return_counts=True)
        baseline = values[np.argmax(counts)]
        # baseline - waveform is positive going signal typically around 0 when there is no signal
        # threshold is the minimum positive signal above baseline

        below = (waveform[0] - baseline) <= threshold
        peak_timebins = []
        max_val = 0
        max_timebin = -1
        for i in range(len(waveform)):
            if below and (waveform[i] - baseline) > threshold:
                below = False
                max_val = 0
                max_timebin = -1
            if not below:
                if (waveform[i] - baseline) > max_val:
                    max_val = waveform[i] - baseline
                    max_timebin = i
                if (waveform[i] - baseline) <= threshold:
                    below = True
                    # if brb != 3 or len(peak_timebins) == 0 or max_timebin > peak_timebins[-1] + 4:  # eliminate peaks from ringing... (for brb03)
                    if 1 == 1:
                        peak_timebins.append(max_timebin)
        return peak_timebins

    def cfd_teststand_method(self, adcs):
        # taken from the process_parquet test stand method
        # Use a cfd like algorithm with delay d = 2, multiplier c = -2
        c = -2.
        d = 2
        # for cfd just use the average of the first 3 adcs as the baseline
        baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.
        # the amplitude is found by adding the highest 3 adcs and subtracting the baseline
        # amp = (baseline - np.min(adcs)) / 100.
        n_largest_vals = sorted(np.array(adcs) - baseline, reverse=True)[:3]
        amp = sum(n_largest_vals)
        # converting to positive going pulses
        data = [(adcs[i] - baseline) + c * (adcs[i - d] - baseline) for i in range(d, len(adcs))]
        # find largest swing zero crossing
        max_diff = 0
        i_md = -1
        for iv in range(1, len(data)):
            if data[iv - 1] > 0. and data[iv] < 0.:
                if data[iv - 1] - data[iv] > max_diff:
                    max_diff = data[iv - 1] - data[iv]
                    i_md = iv

        if i_md > -1:
            x0 = i_md - 1
            y0 = data[i_md - 1]
            x1 = i_md
            y1 = data[i_md]

            # using a linear interpolation, find the value of x for which y = 0
            x = x0 - (x1 - x0) / (y1 - y0) * y0
            # apply offset assuming sigma = 0.96 (see try_cfd.ipynb)
            # x -= 0.5703
            # apply a correction
            apply_correction = True
            offset = 5.
            delta = x - offset
            t = None
            if apply_correction:
                if self.cfd_raw_t[0] < delta < self.cfd_raw_t[-1]:
                    correct_t = np.interp(delta, self.cfd_raw_t, self.cfd_true_t)
                    t = offset + correct_t
                elif delta < self.cfd_raw_t[0]:
                    delta += 1
                    if self.cfd_raw_t[0] < delta < self.cfd_raw_t[-1]:
                        correct_t = np.interp(delta, self.cfd_raw_t, self.cfd_true_t)
                        t = offset - 1 + correct_t
                elif delta > self.cfd_raw_t[-1]:
                    delta -= 1
                    if self.cfd_raw_t[0] < delta < self.cfd_raw_t[-1]:
                        correct_t = np.interp(delta, self.cfd_raw_t, self.cfd_true_t)
                        t = offset + 1 + correct_t
            if t is None:
                t = x - 0.5703
                amp = amp / 2.118  # average correction
            else:
                correct_amp = np.interp(correct_t, self.cfd_true_t, self.amp_raw_t)
                amp /= correct_amp

        else:
            t = -999
            amp = -999

        return t, amp, baseline


class WaveformProcessingmPMT:
    # waveform processing methods as used online by the mPMT
    def __init__(self):
        self.cr_lut_data = np.array(
            [0, 0.0016196, 0.0027526, 0.0039219, 0.0051269, 0.0063669, 0.0076411, 0.0089488, 0.010289, 0.011662,
             0.013066, 0.014501, 0.015966, 0.01746, 0.018982, 0.020533, 0.02211, 0.023714, 0.025344, 0.026998, 0.028677,
             0.03038, 0.032105, 0.033853, 0.035622, 0.037411, 0.039221, 0.04105, 0.042897, 0.044763, 0.046645, 0.048544,
             0.050459, 0.052389, 0.054333, 0.05629, 0.058261, 0.060244, 0.062238, 0.064235, 0.066272, 0.068301,
             0.070348, 0.072394, 0.07445, 0.076504, 0.07856, 0.080649, 0.082724, 0.084811, 0.086887, 0.088973, 0.091051,
             0.093146, 0.095243, 0.097326, 0.099396, 0.10146, 0.10353, 0.1056, 0.10765, 0.10969, 0.11172, 0.11375,
             0.11575, 0.11776, 0.11976, 0.12173, 0.12371, 0.12566, 0.12759, 0.12952, 0.13141, 0.1333, 0.13518, 0.13706,
             0.13894, 0.14078, 0.14258, 0.14436, 0.1461, 0.14785, 0.14955, 0.15123, 0.1529, 0.15453, 0.15612, 0.1577,
             0.15923, 0.16076, 0.16225, 0.16369, 0.1651, 0.16651, 0.16791, 0.16923, 0.17053, 0.1718, 0.17305, 0.17426,
             0.17544, 0.17658, 0.17769, 0.17876, 0.1798, 0.1808, 0.18175, 0.18267, 0.18358, 0.18445, 0.18527, 0.18604,
             0.18678, 0.18749, 0.18817, 0.18881, 0.1894, 0.18996, 0.19048, 0.19098, 0.19144, 0.19186, 0.19225, 0.19258,
             0.19286, 0.19311, 0.19333, 0.19352, 0.19367, 0.19377, 0.19384, 0.19388, 0.19389, 0.19387, 0.19379, 0.19368,
             0.19355, 0.19338, 0.19317, 0.19295, 0.19271, 0.19241, 0.19208, 0.19171, 0.19131, 0.1909, 0.19046, 0.18998,
             0.18948, 0.18895, 0.18838, 0.18777, 0.1871, 0.18642, 0.18571, 0.18496, 0.1842, 0.18341, 0.18258, 0.18172,
             0.18084, 0.17994, 0.17902, 0.17806, 0.17707, 0.17605, 0.175, 0.17396, 0.17288, 0.17175, 0.17059, 0.16944,
             0.16825, 0.16704, 0.16579, 0.1645, 0.1632, 0.16188, 0.16053, 0.15916, 0.15778, 0.15638, 0.15493, 0.15349,
             0.15199, 0.15048, 0.14893, 0.14738, 0.14581, 0.14421, 0.1426, 0.14096, 0.1393, 0.13762, 0.13593, 0.13419,
             0.13246, 0.13069, 0.1289, 0.1271, 0.12529, 0.12346, 0.12161, 0.11973, 0.11785, 0.11595, 0.11403, 0.11207,
             0.11011, 0.10815, 0.10615, 0.10415, 0.10213, 0.10008, 0.098036, 0.095948, 0.093845, 0.091725, 0.089602,
             0.087462, 0.085306, 0.083135, 0.08095, 0.078751, 0.076538, 0.074311, 0.072071, 0.069817, 0.067551,
             0.065271, 0.06298, 0.060676, 0.05836, 0.056032, 0.053692, 0.051342, 0.04898, 0.046607, 0.044224, 0.04183,
             0.039426, 0.037013, 0.034589, 0.032156, 0.029714, 0.027263, 0.024804, 0.022336, 0.019859, 0.017375,
             0.014882, 0.012383, 0.0098757, 0.0073615, 0.0048406, 0.002313, 0])

    def calculate_time_cfd(self, wf_samples, fixed_idx_maxampl=None):
        # method for running on a single waveform sample wf_samples
        cfdp_delay = 1
        cfdp_gain_delayed = 2
        time_base_ns = 8

        delayed = wf_samples * 1
        delayed[cfdp_delay:] = wf_samples[:-cfdp_delay]
        delayed *= -cfdp_gain_delayed

        cf_samples = wf_samples + delayed
        if (cf_samples.min() >= 0) or (cf_samples.max() <= 0):
            return None, None, cf_samples, None
        if fixed_idx_maxampl:
            idx_min = fixed_idx_maxampl + 1
            while cf_samples[idx_min] >= 0:
                idx_min += 1
        else:
            idx_min = np.argmin(cf_samples)

        idx = idx_min
        while idx > 0 and cf_samples[idx] < 0:
            idx -= 1

        if idx > 0:
            x1 = idx
            x2 = x1 + 1
            y1 = cf_samples[x1]
            y2 = cf_samples[x2]
            # Linear interpolation                                                                                                                                                                                                                                                
            y_ratio = y1 / (y1 - y2)
            cf_time = x1 + y_ratio  # time in samples                                                                                                                                                                                                                               
            cf_time_ns = cf_time * time_base_ns  # time in ns                                                                                                                                                                                                                       

            # Correction (based on y_ratio)                                                                                                                                                                                                                                       
            cr = y_ratio
            cr_idx_f = cr * (len(self.cr_lut_data) - 1)
            if cr_idx_f - 0.5 == int(cr_idx_f):
                cr_idx_f += 0.001  # Python uses a rounding method called "round half to even,"                                                                                                                                                                                   
            cr_idx = int(np.round(cr_idx_f))
            if cr_idx > len(self.cr_lut_data) - 1:
                cr_idx = len(self.cr_lut_data) - 1
            time_corr_ns = self.cr_lut_data[cr_idx] * time_base_ns  # ime in ns, NOTE: the cr_lut array contains values in samples, so multiply by time_base_ns
            cf_time_ns_corr = cf_time_ns + time_corr_ns  # Time after correction in ns                                                                                                                                                                                             
            cf_time_corr = cf_time + time_corr_ns / time_base_ns  # Time after correction in samples                                                                                                                                                                                
            # return cf_time, cf_time_corr, None, None
            return cf_time, cf_time_corr, cf_samples, idx
        else:
            return None, None, cf_samples, None

    def cfd_vectorized(self, samples):
        """
        Vectorized CFD for multiple waveforms. Should return the same as the above function.
        needs to be run on a prepared array of waveforms with only one hit per waveform
        in the processing code the 'peak pulse' is always in the same sample for every waveform 

        samples: ndarray (n_wf, wf_len)

        Returns:
            cf_time       : ndarray (n_wf,) CFD times in samples
            cf_time_corr  : ndarray (n_wf,) CFD times after LUT correction in samples
            cf_samples    : ndarray (n_wf, wf_len) CF waveforms
            idx           : ndarray (n_wf,) indices of minimum in CF waveform
        """
        # Parameters
        cfdp_delay = 1
        cfdp_gain_delayed = 2
        time_base_ns = 8
        n_wf, wf_len = samples.shape

        # Step 1: delayed waveform
        delayed = np.zeros_like(samples)
        delayed[:, cfdp_delay:] = samples[:, :-cfdp_delay]
        delayed *= -cfdp_gain_delayed

        # Step 2: CF waveform
        cf_samples = samples + delayed

        # Step 3b: vectorized walk-back
        idx_min = np.argmin(cf_samples, axis=1)

        # neg[i,j] = cf_samples[i,j] < 0
        neg = cf_samples < 0

        # ar[j] = sample index
        ar = np.arange(wf_len)

        # mask[i,j] = (j <= idx_min[i]) AND (cf_samples[i,j] >= 0)
        mask = (ar[None, :] <= idx_min[:, None]) & (~neg)

        # Need the last True j for each waveform
        # If none is True → set to 0
        # reverse the mask array along the waveform axis, argmax gives the last True sample in original non-reversed waveform
        # this but since it is reversed cast position back e.g. last true in index 0 becomes index wf_len -1 in reversed and so on
        last_true = wf_len - 1 - mask[:, ::-1].argmax(axis=1)
        x1 = np.where(mask.any(axis=1), last_true, 0)
        x2 = x1 + 1

        # Step 4: Linear interpolation
        y1 = cf_samples[np.arange(n_wf), x1]
        y2 = cf_samples[np.arange(n_wf), x2]
        y_ratio = y1 / (y1 - y2)
        cf_time = x1 + y_ratio
        cf_time_ns = cf_time * time_base_ns

        # Step 5: LUT correction
        cr = y_ratio
        cr_idx_f = cr * (len(self.cr_lut_data) - 1)
        cr_idx_f += (np.isclose(cr_idx_f - 0.5, np.round(cr_idx_f))) * 0.001
        cr_idx = np.clip(np.round(cr_idx_f).astype(int), 0, len(self.cr_lut_data) - 1)

        time_corr_ns = self.cr_lut_data[cr_idx] * time_base_ns
        cf_time_ns_corr = cf_time_ns + time_corr_ns
        cf_time_corr = cf_time + time_corr_ns / time_base_ns

        return cf_time, cf_time_corr, cf_samples, idx_min

    def charge_calculation_mPMT_method(self, wf_samples, peak_sample):
        # for a single waveform wf
        # peak sample is the index of the peak in the waveform
        # this is the value returned by pulse_finding.do_pulse_finding
        charge = np.sum(wf_samples[peak_sample - 5:peak_sample + 2])
        if wf_samples[peak_sample + 2] > 0:
            charge += wf_samples[peak_sample + 2]
        return charge

    def charge_vectorized_mPMT_method(self, samples, peak_position):
        # a for a vector of waveforms with peak sample always at peak position 
        # assumes the peak sample is always in the same position 
        charge = np.sum(samples[:, (peak_position - 5):(peak_position + 2)], axis=1)
        last_sample = samples[:, (peak_position + 2)]
        hit_charge = charge + np.where(last_sample > 0, last_sample, 0)
        return hit_charge
