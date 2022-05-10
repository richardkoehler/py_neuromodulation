"""Module for handling nm_channels."""
from typing import Iterable, Optional, Union

import pandas as pd


_LFP_TYPES = ["seeg", "dbs", "lfp"]  # must be lower-case


def set_channels(
    ch_names: list[str],
    ch_types: list[str],
    reference: Union[list, str] = "default",
    bads: Optional[list[str]] = None,
    new_names: Union[str, list[str]] = "default",
    ecog_only: bool = False,
    used_types: Iterable[str] | str | None = ("ecog", "dbs", "seeg"),
    target_keywords: Iterable[str] | str | None = ("mov", "squared", "label"),
):
    """
    Return dataframe with channel-specific settings in nm_channels format.

    Return an nm_channels dataframe with the columns: "name", "rereference",
    "used", "target", "type", "status", "new_name"]. "name" is set to ch_names,
    "rereference" can be specified individually. "used" is set to 1 for all
    channel types specified in `used_types`, else to 0. "target" is set to 1
    for all channels containing any of the `target_keywords`, else to 0.

    Possible channel types:
    https://github.com/mne-tools/mne-python/blob/6ae3b22033c745cce5cd5de9b92da54c13c36484/doc/_includes/channel_types.rst

    Arguments
    ---------
        ch_names : list
            list of channel names.
        ch_types : list
            list of channel types. Should optimally be of the types: "ECOG",
            "DBS" or "SEEG".
        reference : str | list of str | None, default: 'default'
            re-referencing scheme. Default is "default". This sets ECOG channel
            references to "average" and creates a bipolar referencing scheme
            for LFP/DBS/SEEG channels, where each channel is referenced to
            the adjacent lower channel, split by left and right hemisphere.
            For this, the channel names must contain the substring "_L_" and/or
            "_R_" (lower or upper case). CAVE: Adjacent channels will be
            determined using the sort() function.
        bads :  str | list of str, default: None
            channels that should be marked as bad and not be used for
            average re-referencing etc.
        new_names : list of str | None, default: 'default'
            new channel names that should be used when writing out the
            features and results. Useful when applying re-referencing. Set to
            `None` if no renaming should be performed. 'default' will infer
            channel renaming from re-referencing information. If a list is
            given, it should be in the same order as `ch_names`.
        ECOG_ONLY : boolean, default: False
            if True, set only 'ecog' channel type to used
        used_types : iterable of str | str | None,
            default : ("ecog", "dbs", "seeg")
            data channel types to be used. Set to `None` to use no channel
            types.
        target_keywords : iterable of str | str| None,
            default : ("ecog", "dbs", "seeg")
            keywords for target channels.

    Returns
    -------
        df: DataFrame in nm_channels format
    """
    if not (len(ch_names) == len(ch_types)):
        raise ValueError(
            "Number of `ch_names` and `ch_types` must match."
            f"Got: {len(ch_names)} `ch_names` and {len(ch_types)} `ch_types`."
        )

    df = pd.DataFrame(
        data=None,
        columns=[
            "name",
            "rereference",
            "used",
            "target",
            "type",
            "status",
            "new_name",
        ],
    )
    df["name"] = ch_names

    if used_types:
        if isinstance(used_types, str):
            used_types = (used_types,)
        used_list = []
        for ch_type in ch_types:
            if any(
                use_type.lower() == ch_type.lower() for use_type in used_types
            ):
                used_list.append(1)
            else:
                used_list.append(0)
        df["used"] = used_list
    else:
        df["used"] = 0

    if target_keywords:
        if isinstance(target_keywords, str):
            target_keywords = (target_keywords,)
        targets = []
        for ch_name in ch_names:
            if any(kw.lower() in ch_name.lower() for kw in target_keywords):
                targets.append(1)
            else:
                targets.append(0)
        df["target"] = targets
    else:
        df["target"] = 0

    # note: BIDS types are in caps, mne.io.RawArray types lower case
    # so that 'type' will be in lower case here
    df["type"] = ch_types

    if ecog_only:
        df.loc[(df["type"] == "seeg") | (df["type"] == "dbs"), "used"] = 0

    if isinstance(reference, str):
        if reference.lower() == "default":
            df = _get_default_references(
                df=df, ch_names=ch_names, ch_types=ch_types
            )
        else:
            raise ValueError(
                "`reference` must be either `default`, `None` or "
                "an iterable of new reference channel names. "
                f"Got: {reference}."
            )

    elif isinstance(reference, list):
        if len(ch_names) != len(reference):
            raise ValueError(
                "Number of `ch_names` and `reference` must match."
                f"Got: {len(ch_names)} `ch_names` and {len(reference)}"
                " `references`."
            )
        df["reference"] = reference
    elif not reference:
        df.loc[:, "rereference"] = "None"
    else:
        raise ValueError(
            "`reference` must be either `default`, None or "
            "an iterable of new reference channel names. "
            f"Got: {reference}."
        )

    if bads:
        if isinstance(bads, str):
            bads = [bads]
        df["status"] = ["bad" if ch in bads else "good" for ch in ch_names]
        df.loc[
            df["status"] == "bad", "used"
        ] = 0  # setting bad channels to not being used
    else:
        df["status"] = "good"

    if not new_names:
        df["new_name"] = ch_names
    elif isinstance(new_names, str):
        if new_names.lower() != "default":
            raise ValueError(
                "`new_names` must be either `default`, None or "
                "an iterable of new channel names. Got: "
                f"{new_names}."
            )
        new_names = []
        for name, ref in zip(df["name"], df["rereference"]):
            if ref == "None":
                new_names.append(name)
            elif ref == "average":
                new_names.append(name + "-avgref")
            else:
                new_names.append(name + "-" + ref)
        df["new_name"] = new_names
    elif hasattr(new_names, "__iter__"):
        if len(new_names) != len(ch_names):
            raise ValueError(
                "Number of `ch_names` and `new_names` must match."
                f" Got: {len(ch_names)} `ch_names` and {len(new_names)}"
                " `new_names`."
            )
        else:
            df["new_name"] = ch_names
    else:
        raise ValueError(
            "`new_names` must be either `default`, None or"
            f" an iterable of new channel names. Got: {new_names}."
        )
    return df


def _get_default_references(
    df: pd.DataFrame, ch_names: list[str], ch_types: list[str]
) -> pd.DataFrame:
    """Add references with default settings (ECOG CAR, LFP bipolar)."""
    ecog_chs = []
    lfp_chs = []
    other_chs = []
    for ch_name, ch_type in zip(ch_names, ch_types):
        if "ecog" in ch_type.lower() or "ecog" in ch_name.lower():
            ecog_chs.append(ch_name)
        elif any(
            lfp_type in ch_type.lower() or lfp_type in ch_name.lower()
            for lfp_type in _LFP_TYPES
        ):
            lfp_chs.append(ch_name)
        else:
            other_chs.append(ch_name)
    lfp_l = [
        lfp_ch
        for lfp_ch in lfp_chs
        if ("_l_" in lfp_ch.lower()) or ("_left_" in lfp_ch.lower())
    ]
    lfp_l.sort()
    lfp_r = [
        lfp_ch
        for lfp_ch in lfp_chs
        if ("_r_" in lfp_ch.lower()) or ("_right_" in lfp_ch.lower())
    ]
    lfp_r.sort()
    lfp_l_refs = [
        lfp_l[i - 1] if i > 0 else lfp_l[-1] for i, _ in enumerate(lfp_l)
    ]
    lfp_r_refs = [
        lfp_r[i - 1] if i > 0 else lfp_r[-1] for i, _ in enumerate(lfp_r)
    ]
    ref_idx = list(df.columns).index("rereference")
    for ecog_ch in ecog_chs:
        df.iloc[df[df["name"] == ecog_ch].index[0], ref_idx] = "average"
    for i, lfp in enumerate(lfp_l):
        df.iloc[df[df["name"] == lfp].index[0], ref_idx] = lfp_l_refs[i]
    for i, lfp in enumerate(lfp_r):
        df.iloc[df[df["name"] == lfp].index[0], ref_idx] = lfp_r_refs[i]
    for other_ch in other_chs:
        df.iloc[df[df["name"] == other_ch].index[0], ref_idx] = "None"
    return df
