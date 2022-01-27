# Import packages and functions
import numpy as np


"""
Functions to re-reference ECoG and LFP signals.
Some relevant literature regarding rereferencing:
    Liu ea, J Neural Eng 2015:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5485665/;
    Dubey, J Neurosc 2019:
    https://www.jneurosci.org/content/39/22/4299

"""


def reref_common_average(data, ch_names):
    """
    Function to perform Common Average Rereferencing.
    Default in ECoG signal processing.
    DISCUSS: Is channel of interest itself included in mean,
    for now: Yes, included in mean.

    Arguments:
        - data (array): 3d array containing time- and signal-
        rows
        - ch_names (list): strings of clean channel names,
        incl 'times', excl removed channels
    
    Returns:
        - newdata (array): identical shape array as input,
        signals re-referenced by extracting the mean over
        all signals for each channel
    """
    newdata = np.empty(data.shape)  # empty array with same shape
    newdata[:, 0, :] = data[:, 0, :]  # copy time rows
    for w in np.arange(data.shape[0]):  # loop over windows
        # take window average of all channels
        ref_mean = np.nanmean(data[w, 1:, :], axis=0)
        for ch in np.arange(1, data.shape[1]):  # loop channels to reref
            newdata[w, ch, :] = data[w, ch, :] - ref_mean
    newnames = [n[:8] for n in ch_names]

    return newdata, newnames


def reref_summing_levels(ch_data, times, chs_og, chs_clean):
    """
    Function to aggregate/sum all segmented electrode contacts
    which belong to same level. The function takes into
    account which contact are empty/bad/removed during arte-
    fact removal.
    TODO: Save printed output into txt-file in derivatives
    
    Arguments:
        - data (array): original signal
        - chs_og (list): all channel names
        - chs_clean (list): ch. names after artef. removal

    Returns:
        - data (array): rereferenced signals
        - side (str): L or R for hemisphere
    """
    try:
        chs_og.remove('time')
    except ValueError:
        print('Sec check: No time row present in orig names')
    side = chs_og[0][4]  # takes 'L' or 'R'

    if chs_og[0][-2] == 'B':  # Boston Scientific leads
        if np.logical_or(len(chs_og) == 15, len(chs_og) == 16):
            # Vercise Cartesia X (15/16 due to ref contact)
            print('Assume BS Vercise Cartesia X')

        contacts = [f'LFP_{side}_{n}_' for n in np.arange(1,17)]
        levels_num = {}
        levels = {}
        for l in np.arange(5):  # Cartesia X: 5 levels w/ 3 contacts
            levels_num[l] = [l * 3, (l * 3) + 1, (l * 3) + 2]
        levels_num[5] = [5 * 3]
        for l in levels_num:
            levels[l] = [contacts[c] for c in levels_num[l]]

        combi_str = '\t'.join(chs_clean)  # pastes all strings to one
        for l in levels:
            for n, c in enumerate(levels[l]):
                # if ch-str is not in combi-str of clean channels
                if c not in combi_str:
                    levels[l].remove(c)
                    levels_num[l].remove(levels_num[l][n])
        
        newdata = np.empty((ch_data.shape[0], len(levels) + 1,
                            ch_data.shape[2]))
        newdata[:, 0, :] = times
        ch_start, ch_stop = 0, 0
        for l in levels_num.keys():
            ch_stop += len(levels_num[l])
            newdata[:, l + 1, :] = np.nanmean(
                ch_data[:, ch_start:ch_stop, :], axis=1
            )
            print(f'Level {side, l} contains rows {ch_start}:{ch_stop}'
                  f', or: {levels[l]}')
            ch_start += len(levels_num[l])

# TODO: CHECK WHERE NAN'S ARE COMING FROM IN CLEANED DATA
    elif chs_og[0][-2:] == 'MT':  # Medtronic leads
        if np.logical_or(len(chs_og) == 7, len(chs_og) == 8):
            print('Assume Medtronic SenSight')

    return newdata, side


def reref_neighb_levels_diff(leveldata, side):
    """
    Function to calculate differences between neighbouring
    eletrode levels. These can come from unsegmented
    electrodes, or from segmented electrodes via the
    reref_summing_levels() function.
    
    Arguments:
        - level (array): 3d array containing [windows,
        time- and level-rows, window_lenght]
        - side(str): L or R for hemisphere

    Returns:
        - newdata (array): rereferenced signals. These
        will contain 1 row less because the differences
        between rows are used
        - sig_names (list): contains corresponnding names
        to the array: 'times', followed by the reref'd
        level-diffs, e.g. 0_1, 3_4
    """
    newdata = np.empty((leveldata.shape[0],
        leveldata.shape[1] - 1, leveldata.shape[2]))
    newdata[:, 0, :] = leveldata[:, 0, :]  # copy times
    sig_names = ['time', ]
    # subtract the level above one level from every level 
    newdata[:, 1:, :] = np.subtract(leveldata[:, 1:-1, :],
                                    leveldata[:, 2:, :])
    # adding signal names
    for level in np.arange(newdata.shape[1] - 1):
        sig_names.append(f'LFP_{side}_{level}_{level + 1}')
        #### TEST!

    return newdata, sig_names


def rereferencing(
    data: dict, group: str,
    ch_names_og=None, ch_names_clean=None
):
    """
    Function to rereference LFP and ECoG data.
    
    Arguments:
        - data: dict containing 3d-arrays per signal
        group (LFP-L/R, ECoG)
        - group: signal group inserted
    
    Returns:
        - datareref: 3d array of rereferenced data of
        inserted signal group
        - names (list): strings of row names, 'times',
        all reref'd signal channels
    """
    print('\n\n======= REREFERENCING OVERVIEW ======')
    data = data[group]  # take group nd-array
    chs_og = ch_names_og[group]  # take group list
    chs_clean = ch_names_clean[group]  # take group list

    if group == 'ecog':
        print(f'\nFor {group}: Common Average Reref')
        rerefdata, names = reref_common_average(
            data=data,
            ch_names=chs_clean)

    elif group[:3] == 'lfp':
        print(f'\nFor {group}: Neigbouring Reref')
        leveldata, side = reref_summing_levels(
            ch_data=data[:, 1:, :],
            times=data[:, 0, :],
            chs_og=chs_og,
            chs_clean=chs_clean,
        )
        rerefdata, names = reref_neighb_levels_diff(
            leveldata=leveldata, side=side,
        )

    # Quality check, delete only nan-channels
    ch_del = []
    for ch in np.arange(rerefdata.shape[1]):
        # check whether ALL values in channel are NaN
        if not (np.isnan(rerefdata[:, ch, :]) == False).any():
            ch_del.append(ch)
    for ch in ch_del:
        rerefdata = np.delete(rerefdata, ch, axis=1)
        print(f'\n Auto Cleaning:\n In {group}: row {ch} ('
            f'{names[ch]}) only contained NaNs and is deleted')
        names.remove(names[ch])


    return rerefdata, names



