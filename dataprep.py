import datetime
import os
import re
from copy import deepcopy
from itertools import product
from numbers import Number
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
from scipy.optimize import curve_fit

import blosc
import dill as pickle
import numpy as np
import pandas as pd
import redis.exceptions
import yaml
from loguru import logger as lg
from tqdm import tqdm
import powerlaw

from constants import *
from trackedcell import TrackedCell


def df_with_dict_cells_to_multiindex(df, n=2):
    dfout = df.stack().to_frame()
    dfout = pd.DataFrame(dfout[0].values.tolist(), index=dfout.index)
    if n > 1:
        dfout = df_with_dict_cells_to_multiindex(dfout, n=n - 1)

    return dfout


def load_config(
        yaml_path: Path or str,
) -> Dict[str, Any]:
    """
    Loads the config file.
    """
    lg.debug(f"Loading config.yaml ...")
    with open(yaml_path) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return cfg


def popul_gen(
        cells: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Tuple[str, str], Dict[str, Any]]:
    """
    Returns a generator that iterates over the cells and treatments and yields
    a tuple of the celltype and treatment and the population dict.
    """
    for ct_ in cells.keys():
        for tr_ in cells[ct_].keys():
            yield (ct_, tr_), cells[ct_][tr_]


def all_cells_have_key(
        cells: Dict[str, Dict[str, Dict[str, Any]]],
        key: str,
) -> bool:
    all_have = True
    for _, popul in popul_gen(cells):
        if key not in popul.keys():
            all_have = False

    return all_have


def fits_from_popul_gen(
        popul: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns a generator that iterates over the fits of the population.
    """
    for datatype_, fit_ in popul['fits'].items():
        if fit_ is not None:
            yield datatype_, fit_


def all_samples_gen(
        cells: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Tuple[str, str], Dict[str, Any]]:
    """
    Returns a generator that iterates over all samples in the dataset.
    """
    for (ct_, tr_), popul in popul_gen(cells):
        for sample_ in popul['samples']:
            yield (ct_, tr_), sample_


def msd_calc(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds


def get_msd(grp):
    msd = pd.DataFrame()

    for i_, grp_ in tqdm(grp):
        if len(grp_) > 1:
            mm = pd.Series(msd_calc(np.array([grp_.x, grp_.y]).T), name=i_)
            msd = pd.concat([msd, mm], axis=1)
        else:
            print("The track is too short to calculate MSD!")

    return msd


def get_aic(loglikeli_sum, n_params):
    aic = - 2 * loglikeli_sum + 2 * n_params
    return aic


def calc_waic(distribs):
    aics = []
    for distrname_, distrib_ in distribs.items():
        distrib_['aic'] = get_aic(distrib_['ll'], distrib_['n_params'])
        aics.append(distrib_['aic'])

    aic_min = min(aics)
    aic_exp_sum = 0

    for distrname_, distrib_ in distribs.items():
        distrib_['aicdiff'] = distrib_['aic'] - aic_min
        aic_exp_sum += np.exp(- distrib_['aicdiff'] / 2)

    for distrname_, distrib_ in distribs.items():
        distrib_['waic'] = np.exp(- distrib_['aicdiff'] / 2) / aic_exp_sum


def dwaics_from_fit(fit):
    try:
        dwaics = val_from_fit(fit, val_name='waic')
    except KeyError:
        calc_waic(fit['distr'])
        dwaics = val_from_fit(fit, val_name='waic')

    return dwaics


def val_from_fit(fit, val_name='waic'):
    vals = {}
    for distrname_, distr_ in fit['distr'].items():
        vals[distrname_] = distr_[val_name]

    return vals


def active_times_and_ratio_from_popul(
        popul: Dict[str, Any],
) -> Dict[str, Number]:
    tdict = {}

    tkeys = [
        k_ for k_ in popul.keys()
        if 'active_t' in k_ or 'total_t' in k_
    ]

    for tk_ in tkeys:
        tdict[tk_] = popul[tk_]

    return tdict


def sample_idx2name(datatype, ct, tr):
    return '{} {} {}'.format(
        datatype_names_more[datatype],
        celltype_names[ct],
        treatment_names[tr],
    )


def get_fits_from_sample(cells, datatype, ct, tr):
    """
    Returns the fits for the given datatype, celltype and treatment.

    :param cells: dict of cells
    :param datatype: str, datatype
    :param ct: str, celltype
    :param tr: str, treatment
    :return: dict of fits
    """
    sample = cells[ct][tr]
    fits = sample['fits']
    return fits[datatype]


def get_fits_waic_df(cells):
    """
    Converts the fits to a dataframe.
    """
    fits = {}
    for (ct_, tr_), popul in popul_gen(cells):
        for datatype_, fit_ in fits_from_popul_gen(popul):
            fits[(datatype_, ct_, tr_)] = dwaics_from_fit(fit_)

    df = pd.DataFrame.from_dict(fits, orient='index')
    df.index.names = waic_index_names_list
    df.sort_index(inplace=True)

    return df


def get_fits_params_df(cells):
    params = {}

    for (ct_, tr_), popul in popul_gen(cells):
        for datatype_, fit_ in fits_from_popul_gen(popul):
            params[(datatype_, ct_, tr_)] = val_from_fit(fit_, val_name='params')

    df = pd.DataFrame.from_dict(params, orient='index')
    df.index.names = waic_index_names_list
    df.sort_index(inplace=True)

    return df


def get_popul_msd_vals_df(
        cells,
        datatype='_as'
):
    vals = {}

    for (ct_, tr_), popul in popul_gen(cells):
        msdkeys = ['msd']
        if 'has_comov' in popul and popul['has_comov']:
            msdkeys.append(f'msd{COMOV_SUFFIX}')
            msdkeys.append(f'msd{NOCOMOV_SUFFIX}')

        for key in msdkeys:
            # Getting list in the tuple so that DataFrame
            # will contain the list within one column.
            vals[(ct_, tr_, key)] = (list(popul[key+datatype].values()),)

    df = pd.DataFrame.from_dict(vals).T
    df = get_comov_treatments_to_multiindex_df(df, 2)
    df = df.droplevel(2)

    if datatype == '_as':
        datatype_keys = ('msd', 'a')
    elif datatype == '_ds':
        datatype_keys = ('msd', 'd')
    else:
        datatype_keys = None

    df.columns = pd.MultiIndex.from_tuples([datatype_keys])

    return df


def get_active_t_vals_df(
        cells,
):
    vals = {}
    ratios = {}
    mean_run = {}
    mean_flight = {}

    for (ct_, tr_), popul in popul_gen(cells):
        ratio_comov = []
        ratio_nocomov = []
        ratio_all = []
        mean_run_l_all = []
        mean_flight_l_all = []
        mean_run_l_comov = []
        mean_flight_l_comov = []
        mean_run_l_nocomov = []
        mean_flight_l_nocomov = []

        for sample in popul['samples']:
            anal = sample['analysis']
            flight_l = anal['flight_l']
            run_l = anal['run_l']
            run_t = anal['run_t']
            total_t = anal['total_t_padded']
            ratio_all.append(sum(anal['run_t']) / total_t)
            mean_run_l_all.append(np.mean(run_l))
            mean_flight_l_all.append(np.mean(flight_l))

            if 'has_comov' in popul and popul['has_comov']:
                flight_l_nocomov = flight_l.copy()
                flight_l_comov = anal[f'flight_l{COMOV_SUFFIX}']
                run_l_nocomov = run_l.copy()
                run_l_comov = anal[f'run_l{COMOV_SUFFIX}']
                run_t_nocomov = run_t.copy()
                run_t_comov = anal[f'run_t{COMOV_SUFFIX}']

                for l_ in flight_l_comov:
                    flight_l_nocomov.remove(l_)

                for l_, t_ in zip(run_l_comov, run_t_comov):
                    run_l_nocomov.remove(l_)
                    run_t_nocomov.remove(t_)

                mean_run_l_comov.append(np.mean(run_l_comov))
                mean_flight_l_comov.append(np.mean(flight_l_comov))
                mean_run_l_nocomov.append(np.mean(run_l_nocomov))
                mean_flight_l_nocomov.append(np.mean(flight_l_nocomov))

                total_t_comov = anal[f'total_t_padded{COMOV_SUFFIX}']
                total_t_nocomov = total_t - total_t_comov
                ratio_comov.append(sum(run_t_comov) / total_t_comov)
                ratio_nocomov.append(sum(run_t_nocomov) / total_t_nocomov)

        vals[(ct_, tr_, 'all')] = {'ratio': ratio_all}
        vals[(ct_, tr_, 'all')]['mean_run_l'] = mean_run_l_all
        vals[(ct_, tr_, 'all')]['mean_flight_l'] = mean_flight_l_all

        if 'has_comov' in popul and popul['has_comov']:
            vals[(ct_, tr_, COMOV_SUFFIX)] = {'ratio': ratio_comov}
            vals[(ct_, tr_, NOCOMOV_SUFFIX)] = {'ratio': ratio_nocomov}

            vals[(ct_, tr_, COMOV_SUFFIX)]['mean_run_l'] = mean_run_l_comov
            vals[(ct_, tr_, COMOV_SUFFIX)]['mean_flight_l'] = mean_flight_l_comov

            vals[(ct_, tr_, NOCOMOV_SUFFIX)]['mean_run_l'] = mean_run_l_nocomov
            vals[(ct_, tr_, NOCOMOV_SUFFIX)]['mean_flight_l'] = mean_flight_l_nocomov

    df = pd.DataFrame.from_dict(vals, orient='index')
    df.columns = pd.MultiIndex.from_product((['active'], df.columns))

    return df


def get_comov_treatments_to_multiindex_df(
        df_input,
        level=0,
):
    df = df_input.copy()

    newidxs1 = []
    newidxs2 = []
    for ind_ in df.index.get_level_values(level):
        if ind_.endswith(COMOV_SUFFIX):
            newidxs1.append(ind_[:-len(COMOV_SUFFIX)])
            newidxs2.append(COMOV_SUFFIX)
        elif ind_.endswith(NOCOMOV_SUFFIX):
            newidxs1.append(ind_[:-len(NOCOMOV_SUFFIX)])
            newidxs2.append(NOCOMOV_SUFFIX)
        else:
            newidxs1.append(ind_)
            newidxs2.append('all')

    # Insert two new levels instead of one initial level
    # between the other existing levels.
    slevels = set(range(df.index.nlevels + 1))
    oldlevels = sorted(slevels.difference([level, level+1]))
    newidxs = [newidxs1, newidxs2]

    new_index = []
    iol_ = 0
    for il_ in range(df.index.nlevels+1):

        if il_ in oldlevels:
            if iol_ == level:
                iol_ += 1
                new_index.append(df.index.get_level_values(iol_))
            else:
                new_index.append(df.index.get_level_values(iol_))

            iol_ += 1

        else:
            new_index.append(newidxs.pop(0))

    new_index = np.array(new_index)
    df.index = pd.MultiIndex.from_arrays(new_index)

    return df


def prepare_fits_waic_df_for_table(
        df,
):
    df = df.copy()

    df.rename(columns=distr_names, inplace=True)
    df.reset_index(inplace=True)

    for idx_ in waic_index_renamings_dict:
        if idx_ in df.columns:
            df[idx_].replace(regex=waic_index_renamings_dict[idx_], inplace=True)

    df['id'] = df.index

    return df


def reindex_fits_df(
        df,
):
    if df.index.nlevels == 4:
        neworder = [0, 2, 3, 1]
    else:
        neworder = [1, 2, 0]
    df_out = df.reorder_levels(neworder).sort_index(axis=0)

    return df_out



def prepare_act_t_df4table(
        df,
):
    df = df.copy()

    df.reset_index(inplace=True)
    df = df.rename(columns={
        'level_0': 'Cell type',
        'level_1': 'Treatment',
        **active_t_ratio_names_more,
    })

    # wAIC index is similar (except the 'Data type'), so we can use
    # the same dict of naming dicts.
    for idx_ in waic_index_renamings_dict:
        if idx_ in df.columns:
            df[idx_].replace(regex=waic_index_renamings_dict[idx_], inplace=True)

    return df


def prepare_msd_df4table(
        df,
):
    df = df.copy()

    df.reset_index(inplace=True)
    df = df.rename(columns={
        'level_0': 'Cell type',
        'level_1': 'Treatment',
        **active_t_ratio_names_more,
    })

    # wAIC index is similar (except the 'Data type'), so we can use
    # the same dict of naming dicts.
    for idx_ in waic_index_renamings_dict:
        if idx_ in df.columns:
            df[idx_].replace(regex=waic_index_renamings_dict[idx_], inplace=True)

    return df


def split_df_by_datatypes(df):
    split_dfs = {}
    grp_waic = df.groupby('Data type')
    for gi_, gg_ in grp_waic:
        for dt_ in datatype_names:
            if dt_ in gi_:
                if dt_ not in split_dfs.keys():
                    split_dfs[dt_] = []
                split_dfs[dt_].append(gg_)

    for dt_, df_ in split_dfs.items():
        df_ = pd.concat(df_)
        split_dfs[dt_] = df_

    return split_dfs


def make_active_t_data(
        sample,
):
    stores = [
        (sample['tc'].active_run_l, sample['tc'].active_run_t, 'run'),
        (sample['tc'].active_flight_l, sample['tc'].active_flight_t, 'flight'),
    ]

    for lstore, tstore, stype in stores:
        ls = []
        ts = []

        for i_ in tstore.keys():
            ls.extend(lstore[i_])
            ts.extend(tstore[i_])

        if 'analysis' not in sample.keys():
            sample['analysis'] = {}

        sample['analysis'][stype + '_l'] = ls
        sample['analysis'][stype + '_t'] = ts

        # Data for lysosomes that for sure have nanoparticles
        if sample['tc'].has_nps:
            ls = []
            ts = []

            for i_ in tstore.keys():
                if i_ in np.unique(sample['tc'].imatch[0]):
                    ls.extend(lstore[i_])
                    ts.extend(tstore[i_])

            sample['analysis'][stype + '_l_np'] = ls
            sample['analysis'][stype + '_t_np'] = ts


def load_separate_summary_files(
        dataset,
        summary_path: Path = Path('out'),
        fileformat='blosc',
):
    cells = {}

    for d_ in dataset:
        cells[d_['celltype']] = {treatment_['treatment']: treatment_
                                 for treatment_ in d_['treatments']}

        for tr_ in cells[d_['celltype']].keys():

            cells[d_['celltype']][tr_]['samples'] = []

            if 'ids' in cells[d_['celltype']][tr_].keys():
                ids2process = cells[d_['celltype']][tr_]['ids']
                sample_sum_files = []
                for id_ in ids2process:
                    sum_file_path_ = (summary_path /
                                      f'{d_["celltype"]}_{tr_}_{id_}_summary.{fileformat}')
                    sample_sum_files.append(sum_file_path_)

            else:
                sample_sum_files = list(
                    summary_path.glob(
                        f'{d_["celltype"]}_{tr_}_*_summary.' + fileformat
                    )
                )

                if 'exclude' in cells[d_['celltype']][tr_].keys():
                    exclude_ids = cells[d_['celltype']][tr_]['exclude']
                    ex_idxs_ = []

                    for ienum_, sum_file_ in enumerate(sample_sum_files):
                        file_id_ = re.search('.+_(.+)(?=_summary)', sum_file_.name)[1]
                        if file_id_ in exclude_ids:
                            ex_idxs_.append(ienum_)

                    sample_sum_files = [
                        sum_file_ for ienum_, sum_file_ in enumerate(sample_sum_files)
                        if ienum_ not in ex_idxs_
                    ]

            for sum_file_ in sample_sum_files:
                with open(sum_file_, 'rb') as fh:
                    lg.debug(f"Loading [{sum_file_}] ...")

                    if fileformat == 'blosc':
                        lg.debug(f"Decompressing [{sum_file_}] ...")
                        bts = blosc.decompress(fh.read())
                        lg.debug(f"Unpickling data ...")
                        cells[d_['celltype']][tr_]['samples'].append(
                            pickle.loads(bts))

                    elif fileformat == 'pkl':
                        lg.debug(f"Loading [{sum_file_}]")
                        cells[d_['celltype']][tr_]['samples'].append(
                            pickle.load(fh))

    return cells


def load_cell_summary(
        name,
        summary_path: Path = Path('out'),
        fileformat='blosc',
):
    sample_file = list(summary_path.glob(
        f'{name}_summary.' + fileformat))[0]

    if sample_file.exists():
        with open(sample_file, 'rb') as fh:
            lg.debug(f"Loading [{sample_file}] ...")

            if fileformat == 'blosc':
                lg.debug(f"Decompressing [{sample_file}] ...")
                bts = blosc.decompress(fh.read())
                lg.debug(f"Unpickling data ...")
                sample = pickle.loads(bts)

            elif fileformat == 'pkl':
                lg.debug(f"Loading [{sample_file}]")
                sample = pickle.load(fh)

        return sample

    else:
        lg.warning(f"No summary file found for [{name}]")


def load_to_cells_summary_redis(cells, redis_serv):
    for _, sample in all_samples_gen(cells):
        cell_sum_name = sample['name'] + '_summary'
        sample_loaded = pickle.loads(redis_serv.get(cell_sum_name))
        if sample_loaded is None:
            lg.warning(f"No summary found for [{cell_sum_name}] in redis."
                       f" Loading from file ...")
            sample_loaded = load_cell_summary(cell_sum_name)

            if sample_loaded is None:
                lg.warning(f"No summary found for [{cell_sum_name}].")
                continue

        sample.update(sample_loaded)


def save_cells_summary_redis(cells, redis_serv):
    for (ct, tr), sample in all_samples_gen(cells):
        cell_sum_name_ = sample['name'] + '_summary'

        if redis_serv.exists(cell_sum_name_):
            lg.warning(f"Redis key {cell_sum_name_} "
                       f"already exists; skipping.")
        else:
            lg.debug(f"Saving {cell_sum_name_} to redis ...")
            try:
                redis_serv.set(
                    cell_sum_name_,
                    pickle.dumps(sample),
                )
            except redis.exceptions.ConnectionError:
                lg.warning(f"Redis connection error. Skipping.")


def load_monosummary(
        summary_dir,
        fileformat='pkl',
):
    summary_dir = Path(summary_dir)

    if fileformat == 'blosc':
        parts = list(summary_dir.glob('cells_summary*.blosc'))
        bts = bytes()

        if len(parts) > 1:
            lg.debug(f"Multiple summary files found in [{summary_dir}].")
            for part in parts:
                with open(part, 'rb') as fh:
                    lg.debug(f"Loading [{part}] ...")
                    comp_bts = fh.read()
                    lg.debug(f"Decompressing the data ...")
                    bts += blosc.decompress(comp_bts)

        else:
            with open(parts[0], 'rb') as fh:
                lg.debug(f"Loading [{fh}] ...")
                bts = fh.read()
                lg.debug(f"Decompressing the data ...")
                bts = blosc.decompress(bts)

        lg.debug(f"Unpickling the data ...")
        cells = pickle.loads(bts)

    else:
        summary_path = summary_dir.joinpath(f'cells_summary.pkl')
        with open(summary_path, 'rb') as fh:
            lg.debug(f"Loading [{summary_path}] ...")
            cells = pickle.load(fh)

    lg.debug(f"Done loading the summary file.")

    return cells


def save_cell_summary(
        sample,
        cell_sum_name,
        fileformat='blosc',
        sum_dir: Path = Path('out'),
        rewrite=False,
):
    sum_path = Path(sum_dir, cell_sum_name)

    if sum_path.exists():
        if not rewrite:
            lg.debug(f"File [{sum_path}] already exists; skipping.")
            return
        else:
            lg.debug(f"File [{sum_path}] already exists; re-writing it.")

    with open(sum_path, 'wb') as fh:
        lg.debug(f"Saving [{cell_sum_name}] ...")
        if fileformat == 'blosc':
            lg.debug(f"Compressing {cell_sum_name[:-len(fileformat) - 1]} ...")
            bts = pickle.dumps(sample)
            lg.debug(f"Writing to file ...")
            fh.write(blosc.compress(bts))
        elif fileformat == 'pkl':
            pickle.dump(sample, fh)


def save_cells_summary_separate(
        cells,
        fileformat='blosc',
        sum_dir: Path = Path('out'),
        rewrite=False,
):
    for ct_ in cells.values():
        for tr_ in ct_.values():
            for s_ in tr_['samples']:
                cell_sum_name = s_['tc'].name + '_summary.' + fileformat
                save_cell_summary(
                    s_,
                    cell_sum_name,
                    fileformat,
                    sum_dir,
                    rewrite=rewrite,
                )


def split_monosummary(
        summary_path,
        in_fileformat='pkl',
        out_fileformat='blosc',
):
    cells = load_monosummary(summary_path, fileformat=in_fileformat)
    save_cells_summary_separate(cells, fileformat=out_fileformat)


def make_monosummary(
        cells,
        out_dir='out',
        fileformat='pkl',
        partsize=int(2e9),
):
    out_dir = Path(out_dir)
    out_path = out_dir / ('cells_summary.' + fileformat)

    if fileformat == 'blosc':
        lg.debug(f"Serializing the summary ...")
        bts = pickle.dumps(cells)

        l_ = len(bts)

        if l_ > partsize:
            nparts = l_ // partsize + 1
            lg.debug(f"Saving the summary in {nparts} parts.")

            for i in range(nparts):
                out_path = out_dir / f'cells_summary_part{i + 1}.{fileformat}'

                lg.debug(f"Saving part {i + 1}.")
                lg.debug(f"Compressing the summary ...")

                bts_part = blosc.compress(
                    bts[i * partsize:(i + 1) * partsize])

                lg.debug(f"Writing to file [{out_path}] ...")
                with open(out_path, 'wb') as fh:
                    fh.write(bts_part)

        else:
            lg.debug(f"Saving the summary in one part.")

            lg.debug(f"Compressing the summary ...")
            bts = blosc.compress(bts)

            lg.debug(f"Writing to file [{out_path}] ...")
            with open(out_path, 'wb') as fh:
                fh.write(bts)

    elif fileformat == 'pkl':
        lg.debug(f"Writing to file [{out_path}] ...")
        with open(out_path, 'wb') as fh:
            pickle.dump(cells, fh)

    lg.debug(f"Done saving the summary.")


def load_batch(
        dataset,
        process_data=False,
        savefile4parsed_data=None,
        do_msd=True,
        out_dir='out',
):
    cells = {}

    if savefile4parsed_data:
        savefile4parsed_data = Path(savefile4parsed_data)

        if savefile4parsed_data.is_file() and savefile4parsed_data.exists():
            filepath_ = savefile4parsed_data
            modified_time_ = os.path.getmtime(filepath_)
            timestamp_ = datetime.datetime.fromtimestamp(modified_time_) \
                .strftime("%Y-%b-%d_%H-%M-%S")

            lg.debug(f"Backing up the file with parsed"
                     f" data info: [{str(filepath_) + '_' + timestamp_}]")
            os.rename(filepath_, str(filepath_) + '_' + timestamp_)

    for d_ in dataset:
        cells[d_['celltype']] = {treatment_['treatment']: treatment_
                                 for treatment_ in d_['treatments']}

        for tr_ in cells[d_['celltype']].keys():
            dir_ = Path(cells[d_['celltype']][tr_]['dir'])

            if not dir_.exists():
                lg.warning(f"[{d_['celltype']}|{tr_}]:"
                           f" Directory {dir_} does not exist; skipping.")

            else:
                lg.debug(f"[{d_['celltype']}|{tr_}]:"
                         f" Loading data from {dir_} ...")
                cells[d_['celltype']][tr_]['samples'] = []

                if 'ids' in cells[d_['celltype']][tr_].keys():
                    ids2process = cells[d_['celltype']][tr_]['ids']
                else:
                    ids2process = None

                for file_ in dir_.glob('*.csv'):
                    fname_regex = re.compile(r'(\d+) (.*?) (....s?) +?')
                    fname_match = fname_regex.match(file_.name)

                    if fname_match is None:
                        lg.warning(f"File name does not match expected pattern: [{file_.name}],"
                                   f" trying a pattern without the treatment label.")
                        fname_regex = re.compile(r'(\d+) (.*?) +?')
                        fname_match = fname_regex.match(file_.name)

                        if fname_match is None:
                            lg.warning(f"File name does not match expected format: [{file_}];"
                                       f" skipping.")
                            continue

                    regex_grps = fname_match.groups()

                    celltype_ = regex_grps[1].lower()

                    if len(fname_match.groups()) == 2:
                        treatment_ = tr_
                    else:
                        treatment_ = regex_grps[2].lower()[:4]

                    if (len(regex_grps) != 3 and len(regex_grps) != 2) \
                            or celltype_ != d_['celltype'] \
                            or treatment_ not in cells[d_['celltype']]:

                        lg.warning(f"File name does not match expected format: [{file_}];"
                                   f" skipping.")

                    else:
                        cname_ = f"{celltype_}_{treatment_}_{regex_grps[0]}"

                        if (not ids2process) \
                                or (ids2process and regex_grps[0] in ids2process):

                            if savefile4parsed_data:
                                lg.debug(f"Saving parsed data to [{savefile4parsed_data}] ...")
                                with open(savefile4parsed_data, 'a+') as fparsed:
                                    fparsed.write(
                                        f'{str(file_)}, {cname_}\n'
                                    )

                            sample_ = dict()
                            sample_['id'] = regex_grps[0]
                            sample_['name'] = cname_
                            sample_['tracks_file'] = str(file_)

                            if process_data:
                                lg.debug(f"[{d_['celltype']}|{tr_}]:"
                                         f" Loading data from [{file_}] ...")

                                has_nps = treatment_ != 'nonp'

                                sample_['tc'] = TrackedCell(
                                    name=cname_,
                                    filepath=file_,
                                    has_nps=has_nps,
                                )

                                sample_['tc'].get_active_runs()
                                sample_['tc'].get_active_flights()
                                make_active_t_data(sample_)

                                if do_msd:
                                    savefile_msd = Path(out_dir, f"{cname_}_msd.pkl")
                                    if 'analysis' not in sample_.keys():
                                        sample_['analysis'] = dict()

                                    if savefile_msd.exists():
                                        lg.debug(f"Loading MSD data from [{savefile_msd}] ...")
                                        with open(savefile_msd, 'rb') as fmsd:
                                            sample_['analysis']['msd'] = pickle.load(fmsd)

                                    else:
                                        df_ = sample_['tc'].df_lys
                                        df_ = df_.rename(columns={'posx': 'x', 'posy': 'y'})
                                        grp_ = df_.groupby('id')
                                        lg.debug(f"Calculating MSD for [{cname_}] ...")
                                        sample_['analysis']['msd'] = get_msd(grp_)

                                        with open(savefile_msd, 'wb+') as fmsd:
                                            lg.debug(f"Saving MSD data to [{savefile_msd}] ...")
                                            pickle.dump(sample_['analysis']['msd'], fmsd)

                            cells[d_['celltype']][tr_]['samples'].append(sample_)

                        else:
                            lg.warning(f"Sample id={regex_grps[0]} is not in the dataset; skipping.")

    return cells


def get_data_ccdf(data):
    xs = list(sorted(set(data)))
    num_elements = [data.count(num_el) for num_el in xs]
    ccdf_notnorm = []
    for i, v in enumerate(num_elements):
        ccdf_notnorm.append(np.cumsum(num_elements[i:])[-1])
    ccdf = [float(x) / ccdf_notnorm[0] for x in ccdf_notnorm]
    return xs, ccdf


def get_singlecell_cdf_df(
        cells: Dict,
        celltypes: List,
        treatments: List,
        data_type='dir_t',
):
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_data_dict = {}
    df_cdf_dict = {}
    for ct_, tr_ in product(celltypes, treatments):
        for cell_ in cells[ct_][tr_]['samples']:
            data_ = cell_['analysis']['ddf_turn_ang'][120]['res'][data_type]

            df_cdf_dict[(ct_, tr_, cell_['id'])] = pd.DataFrame(
                get_data_ccdf(data_.to_list())
            ).T

            df_data_dict[(ct_, tr_, cell_['id'])] = data_

    df_data = pd.concat(df_data_dict)
    df_cdf = pd.concat(df_cdf_dict)

    return df_cdf, df_data


def get_summary_cdf_df(
        cells: Dict,
        celltypes: List,
        treatments: List,
        data_type='dir_t',
):
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_data_dict = {}
    df_cdf_dict = {}

    for ct_, tr_ in product(celltypes, treatments):
        ldata = []

        for cell_ in cells[ct_][tr_]['samples']:
            ldata.extend(
                cell_['analysis']['ddf_turn_ang'][120]
                ['res'][data_type].to_list()
            )

        df_cdf_dict[(ct_, tr_)] = pd.DataFrame(
            get_data_ccdf(ldata)
        ).T

        df_data_dict[(ct_, tr_)] = ldata

    df_cdf = pd.concat(df_cdf_dict)

    return df_cdf


def get_act_cdf_df(
        cells: Dict,
        celltypes: List,
        treatments: List,
        data_type='active_t',
        from_summary=False,
):
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_cdf_dict = {}
    data_dict = {}

    if (celltypes is None) or (treatments is None):
        return None

    if from_summary:
        for ct, tr in product(celltypes, treatments):
            data = get_data_ccdf_from_fitobj(
                cells, ct, tr, data_type)

            df_cdf_dict[(ct, tr)] = pd.DataFrame(data).T

            if 'has_comov' in cells[ct][tr] and cells[ct][tr]['has_comov']:
                data = get_data_ccdf_from_fitobj(
                    cells, ct, tr, data_type+COMOV_SUFFIX)

                df_cdf_dict[(ct, tr+COMOV_SUFFIX)] = pd.DataFrame(data).T

            ldata = []
            if not tr in cells[ct]:
                continue

            for cell_ in cells[ct][tr]['samples']:
                ldata.extend(cell_['analysis'][data_type])

            data_dict[(ct, tr)] = ldata

    else:
        for ct, tr in product(celltypes, treatments):
            ldata = []

            if tr in cells[ct]:
                for cell_ in cells[ct][tr]['samples']:
                    ldata.extend(cell_['analysis'][data_type])

            df_cdf_dict[(ct, tr)] = pd.DataFrame(
                get_data_ccdf(ldata)
            ).T

            data_dict[(ct, tr)] = ldata

            if tr != 'nonp':
                ldata = []

                if tr in cells[ct]:
                    for cell_ in cells[ct][tr]['samples']:
                        ldata.extend(cell_['analysis'][data_type + '_np'])

                df_cdf_dict[(ct, tr + '_np')] = pd.DataFrame(
                    get_data_ccdf(ldata)
                ).T

                data_dict[(ct, tr + '_np')] = ldata

    df_cdf = pd.concat(df_cdf_dict)

    return df_cdf, data_dict


def fit_longtail(
        data,
        xmin=None,
        float_precision=6,
):
    data_np = np.array(data)

    if data_np.dtype == np.float:
        data_np = np.around(data_np, decimals=float_precision)

    if xmin is None:
        xmin = data_np.min() - 1e-12
    elif xmin == 0:
        xmin = np.unique(data_np[data_np > 0]).min()

    data_np = np.delete(data_np, np.where(data_np < xmin))

    plfit = powerlaw.Fit(data_np, xmin=xmin, discrete=False)
    plfit.power_law.loglikelihood = plfit.power_law.loglikelihoods(data_np).sum()
    plfit.power_law.parameter1_name = 'alpha'
    plfit.power_law.parameter1 = plfit.alpha

    return plfit


def fit_data(cells):
    for ct_ in cells.keys():
        for tr_ in cells[ct_].keys():
            ddata = {}
            samples_ = cells[ct_][tr_]['samples']
            for s_ in samples_:
                for an_, data_ in s_['analysis'].items():
                    if ('run_' in an_) or ('flight_' in an_):
                        if an_ not in ddata.keys():
                            ddata[an_] = []

                        ddata[an_].extend(data_)

            dfits = {}
            for an_, data_ in ddata.items():
                if ('run_' in an_) or ('flight_' in an_):
                    dfits[an_] = {}
                    dfits[an_]['fitobj'] = fit_longtail(
                        data_, xmin=np.min(data_))

            cells[ct_][tr_]['fits'] = dfits


def collect_all_msds(cells):
    lg.debug(f"Collecting all MSDs ...")
    for (ct_, tr_), popul in popul_gen(cells):
        all_dfs = []
        np_dfs = []

        for cell_ in cells[ct_][tr_]['samples']:
            msd_all = cell_['analysis']['msd']

            if not isinstance(msd_all.columns, pd.MultiIndex):
                col_multiidx = pd.MultiIndex.from_product(
                    [[cell_['id']], msd_all.columns]
                )
                msd_all.columns = col_multiidx

            all_dfs.append(msd_all)

            if tr_ in cells[ct_] and tr_ != 'nonp':
                msd_np = msd_all.iloc[:,
                         np.unique(cell_['tc'].imatch[0])]
                np_dfs.append(msd_np)

        msd = pd.concat(all_dfs, axis=1)
        popul['msd'] = msd

        if tr_ in cells[ct_] and tr_ != 'nonp':
            popul['has_comov'] = True
            msd_np = pd.concat(np_dfs, axis=1)
            popul['msd' + COMOV_SUFFIX] = msd_np


def collect_samples_msds(popul):
    msdkeys = ['msd']
    if 'has_comov' in popul and popul['has_comov']:
        msdkeys.append('msd' + COMOV_SUFFIX)

    for sample_ in popul['samples']:
        id_ = sample_['id']
        for msdkey in msdkeys:
            popul['samples_' + msdkey][id_] = popul[msdkey].loc[:, id_].mean(axis=1)


def fit_msd(timedata, msddata):
    def func2fit(x, d, a):
        return 4 * d * (x ** a)

    values, cov = curve_fit(func2fit, timedata, msddata)
    return values, cov


def get_nocomov_msd_fits(
        popul,
        fitlims=(2, 20),
):
    if not ('has_comov' in popul and popul['has_comov']):
        return

    nocomov_cols = popul['msd'].columns.difference(popul['msd_np'].columns)
    nocomov_msd = popul['msd'].loc[:, nocomov_cols]
    popul[f'msd{NOCOMOV_SUFFIX}_as'] = {}
    popul[f'msd{NOCOMOV_SUFFIX}_ds'] = {}
    for i_, g_ in nocomov_msd.groupby(level=0, axis=1):
        msd_ = g_.iloc[fitlims[0]:fitlims[1]].mean(axis=1)
        fit_ = fit_msd(
            msd_.index.values / FPS,
            msd_.values,
            )[0]

        popul[f'msd{NOCOMOV_SUFFIX}_as'][i_] = fit_[1]
        popul[f'msd{NOCOMOV_SUFFIX}_ds'][i_] = fit_[0]


def calc_avg_msd_curves(cells):
    for _, popul in popul_gen(cells):
        msdkeys = ['msd']
        if 'has_comov' in popul and popul['has_comov']:
            msdkeys.append('msd' + COMOV_SUFFIX)

        for msdkey in msdkeys:
            avg_msd = pd.concat(
                [msd_ for msd_ in popul['samples_'+msdkey].values()],
                axis=1
            ).mean(axis=1)
            popul[f'avg_{msdkey}_curve'] = avg_msd


def fit_msd_each_cell_from_raw(
        cells,
        fitlims=(0, 20),
):
    for (ct, tr), sample in tqdm(all_samples_gen(cells)):
        msd = sample['analysis']['msd'].loc[:, sample['id']]
        msdmean = msd.iloc[fitlims[0]:fitlims[1]].mean(axis=1)
        sample['analysis']['msd_fit'] = fit_msd(
            msdmean.index.values / FPS,
            msdmean.values,
            )

        tc: TrackedCell
        tc = sample['tc']

        if tc.has_nps:
            ids = tc.df_lys.index.get_level_values(0).unique()
            comov_ids = ids[np.unique(tc.imatch[0])]
            msdmean = msd.loc[:, comov_ids].iloc[fitlims[0]:fitlims[1]].mean(axis=1)
            sample['analysis'][f'msd{COMOV_SUFFIX}_fit'] = fit_msd(
                msdmean.index.values / FPS,
                msdmean.values,
                )


def fit_msd_each_cell_from_anal_summary(
        cells,
        fitlims=(0, 20),
):
    msdkeys_base = ['msd', ]
    msd_comov_suf = f'msd{COMOV_SUFFIX}'

    for (ct, tr), popul in popul_gen(cells):
        msdkeys = msdkeys_base.copy()
        if 'has_comov' in popul and popul['has_comov']:
            msdkeys.append(msd_comov_suf)

        for msdkey in msdkeys:
            for sample in popul['samples']:
                msd_ = popul['samples_'+msdkey][sample['id']]
                msd_ = msd_.iloc[fitlims[0]:fitlims[1]]

                sample['analysis'][msdkey+'_fit'] = fit_msd(
                    msd_.index.values / FPS,
                    msd_.values,
                    )


def fit_msd_population(
        cells,
        fitlims=(0, 20),
        collect_msds=True,
):
    if collect_msds:
        collect_all_msds(cells)

    lg.debug(f"Fitting MSDs ...")
    for _, popul in popul_gen(cells):
        msd = popul['msd']
        msdmean = msd.iloc[fitlims[0]:fitlims[1]].mean(axis=1)
        popul['msd_fit'] = fit_msd(
            msdmean.index.values / FPS,
            msdmean.values,
            )

        if 'msd' + COMOV_SUFFIX in popul.keys():
            msd = popul['msd' + COMOV_SUFFIX]
            msdmean = msd.iloc[fitlims[0]:fitlims[1]].mean(axis=1)
            popul[f'msd{COMOV_SUFFIX}_fit'] = fit_msd(
                msdmean.index.values / FPS,
                msdmean.values,
                )


def calc_mean_population_msd(cells):
    for _, popul in popul_gen(cells):
        if 'means' not in popul:
            popul['means'] = {}

        if 'stds' not in popul:
            popul['stds'] = {}

        msdkeys = ['msd', ]
        msd_comov_suf = f'msd{COMOV_SUFFIX}'

        if f'msd{COMOV_SUFFIX}_fit' in popul['samples'][0]['analysis']:
            popul['has_comov'] = True
            msdkeys.append(msd_comov_suf)

        for msdkey_ in msdkeys:
            popul['means'][msdkey_] = {}
            popul['stds'][msdkey_] = {}
            if 'samples_' + msdkey_ not in popul:
                popul['samples_' + msdkey_] = {}
            popul[msdkey_ + '_as'] = {}
            popul[msdkey_ + '_ds'] = {}

        for sample in popul['samples']:
            msdkeys = ['msd', ]
            id_ = sample['id']
            anal = sample['analysis']
            if f'msd{COMOV_SUFFIX}_fit' in anal.keys():
                msdkeys.append(msd_comov_suf)

            if 'samples_msd' not in popul:
                popul['samples_msd'][id_] = anal['msd'].mean(axis=1)

            if 'has_comov' in popul and popul['has_comov']:
                if not 'samples_'+msd_comov_suf in popul:
                    idxs_ = np.unique(sample['tc'].imatch[0])
                    idxs_ = sample['tc'].df_lys.index.get_level_values(0).unique()[idxs_]
                    msd_comov = anal['msd'].loc[:, idxs_].mean(axis=1)
                    popul['samples_' + msd_comov_suf][id_] = msd_comov

            for msdkey_ in msdkeys:
                popul[msdkey_ + '_as'][id_] = anal[msdkey_ + '_fit'][0][1]
                popul[msdkey_ + '_ds'][id_] = anal[msdkey_ + '_fit'][0][0]

        msdkeys = ['msd', ]
        if 'has_comov' in popul and popul['has_comov']:
            msdkeys.append(msd_comov_suf)

        for msdkey_ in msdkeys:
            msd_as_ = np.array(list(popul[msdkey_ + '_as'].values()))
            msd_ds_ = np.array(list(popul[msdkey_ + '_ds'].values()))

            popul['means'][msdkey_]['a'] = msd_as_.mean()
            popul['means'][msdkey_]['d'] = msd_ds_.mean()
            popul['stds'][msdkey_]['a'] = msd_as_.std()
            popul['stds'][msdkey_]['d'] = msd_ds_.std()


def calc_mean_population_act_t(cells):
    for _, popul in popul_gen(cells):
        if 'means' not in popul:
            popul['means'] = {}

        if 'stds' not in popul:
            popul['stds'] = {}

        actkey = 'active_t'
        actkeys_base = [actkey, ]
        actkeys = actkeys_base.copy()
        act_t_comov_suf = f'{actkey}{COMOV_SUFFIX}'

        if f'{actkey}{COMOV_SUFFIX}' in popul['samples'][0]['analysis']:
            popul['has_comov'] = True
            actkeys.append(act_t_comov_suf)

        for key_ in actkeys:
            popul['means'][key_] = {}  # 'ratio': {}}
            popul['stds'][key_] = {}  # 'ratio': {}}
            popul['samples_' + key_] = {'ratio': {}}

        for sample in popul['samples']:
            id_ = sample['id']
            anal = sample['analysis']

            popul['samples_' + actkey]['ratio'][id_] = \
                anal[actkey + '_ratio']

            if 'has_comov' in popul and popul['has_comov']:
                act_ratio_comov_key = 'active_t_ratio' + COMOV_SUFFIX
                popul['samples_' + act_t_comov_suf]['ratio'][id_] = \
                    anal[act_ratio_comov_key]

        actkeys = actkeys_base.copy()

        if 'has_comov' in popul and popul['has_comov']:
            actkeys.append(act_t_comov_suf)

        for key_ in actkeys:
            values_ = np.array(list(popul['samples_' + key_]['ratio'].values()))

            popul['means'][key_]['ratio'] = values_.mean()
            popul['stds'][key_]['ratio'] = values_.std()


def calc_active_time_ratio(
        cells,
        suffix: str or Union[List, Tuple] = ('', COMOV_SUFFIX),
) -> None:
    """
    Calculate the ratio of active time to total time for each cell type and
    treatment.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    for (ct_, tr_), sample in all_samples_gen(cells):
        for suff_ in suffix:
            if suff_ == COMOV_SUFFIX:
                if sample['tc'].imatch is not None:
                    idxs_ = np.unique(sample['tc'].imatch[0])
                    idxs_ = sample['tc'].df_lys.index.get_level_values(0).unique()[idxs_]
                    df_ = sample['tc'].df_lys.loc[idxs_]
                    grp_ = df_.reset_index(level=1).groupby('id')

                else:
                    continue

            elif suff_ == '':
                grp_ = sample['tc'].df_lys.reset_index(level=1).groupby('id')

            # If the suffix is unknown, do nothing.
            else:
                continue

            # Since active time analysis does not count the first
            # and last wavelet padding (WV_PAD) frames, we need to
            # exclude those frames from the total time.

            if 'total_t_padded' + suff_ not in cells[ct_][tr_].keys():
                cells[ct_][tr_]['total_t_padded' + suff_] = 0
            if 'active_t' + suff_ not in cells[ct_][tr_].keys():
                cells[ct_][tr_]['active_t' + suff_] = 0

            ttot_ = np.sum(grp_.size() - 2 * WV_PAD)
            sample['analysis']['total_t_padded' + suff_] = ttot_
            cells[ct_][tr_]['total_t_padded' + suff_] += ttot_

            tact_ = np.sum(sample['analysis']['run_t' + suff_])
            sample['analysis']['active_t' + suff_] = tact_
            cells[ct_][tr_]['active_t' + suff_] += tact_

            sample['analysis']['active_t_ratio' + suff_] = tact_ / ttot_

    for _, popul in popul_gen(cells):
        for suff_ in suffix:

            if suff_ in ['', COMOV_SUFFIX] and \
                    'total_t_padded' + suff_ in popul.keys():
                popul['active_t_ratio' + suff_] = \
                    popul['active_t' + suff_] / popul['total_t_padded' + suff_]


def get_active_time_ratio_df(
        cells,
):
    """
    Return a dataframe with the ratio of active time to total time for each
    cell type and treatment.
    """
    if not all_cells_have_key(cells, 'active_t_ratio'):
        lg.debug(f"Not all cell populations have active time ratio calculated;"
                 f" calculating ...")
        calc_active_time_ratio(cells)

    ddf = {}
    for pgo_ in popul_gen(cells):
        idx_, popul = pgo_

        ddf[idx_] = active_times_and_ratio_from_popul(popul)

    df = pd.DataFrame.from_dict(ddf, orient='index')

    return df


def get_msd_df(cells):
    """
    Return a dataframe with the mean square displacement for each cell type and
    treatment.
    """
    ddf = {}
    for pgo_ in popul_gen(cells):
        idx_, popul = pgo_

        pmeans = popul['means']

        pdict = {}
        for k, v in pmeans.items():
            if 'msd' in k:
                pdict[k] = v['msd']

        ddf[idx_] = 'dd'

    df = pd.DataFrame.from_dict(ddf, orient='index')

    return df


def popul_means_to_df(
        cells,
        key1,
        key2,
):
    dfdata = {}

    for (ct, tr), popul in popul_gen(cells):
        if tr not in dfdata:
            dfdata[tr] = {}

        dfdata[tr][ct] = popul['means'][key1][key2]

        if 'has_comov' in popul and popul['has_comov']:
            if tr + COMOV_SUFFIX not in dfdata:
                dfdata[tr + COMOV_SUFFIX] = {}

            dfdata[tr + COMOV_SUFFIX][ct] = \
                popul['means'][key1 + COMOV_SUFFIX][key2]

    df = pd.DataFrame.from_dict(dfdata, orient='index')
    return df


def popul_stats_to_df(
        cells,
        key1,
        key2,
):
    dfdata = {}
    lstats = ['means', 'stds']

    for (ct, tr), popul in popul_gen(cells):
        if tr not in dfdata:
            dfdata[tr] = {}

        if ct not in dfdata[tr]:
            dfdata[tr][ct] = {}
            dfdata[tr][ct]['all'] = {}

        dpop = dfdata[tr][ct]

        for kstat in lstats:
            if kstat not in dfdata[tr][ct]['all']:
                dpop['all'][kstat] = {}

            dpop['all'][kstat] = popul[kstat][key1][key2]

            if 'has_comov' in popul and popul['has_comov']:
                if COMOV_SUFFIX not in dpop:
                    dpop[COMOV_SUFFIX] = {}
                dpop[COMOV_SUFFIX][kstat] = popul[kstat][key1 + COMOV_SUFFIX][key2]

                if key1 + NOCOMOV_SUFFIX in popul[kstat]:
                    if NOCOMOV_SUFFIX not in dpop:
                        dpop[NOCOMOV_SUFFIX] = {}
                    dpop[NOCOMOV_SUFFIX][kstat] = popul[kstat][key1 + NOCOMOV_SUFFIX][key2]

    df = pd.DataFrame(dfdata)
    dfout = df_with_dict_cells_to_multiindex(df, n=2)

    ncols = len(dfout.columns)
    hier_cols = pd.MultiIndex.from_arrays(
        [[key1] * ncols, [key2] * ncols, dfout.columns])
    dfout.columns = hier_cols

    return dfout


def save_cells_analysis_summary(
        cells,
        sum_dir: Path = Path('out'),
        rewrite=False,
):
    fpath = sum_dir / 'cells_analysis_summary_flightl.pkl'
    lg.debug(f"Saving cells analysis summary to {fpath} ...")
    sum_dir = Path(sum_dir)

    dout = {}

    for (ct, tr), popul in popul_gen(cells):
        if ct not in dout:
            dout[ct] = {}

        dout[ct][tr] = {'samples': []}

        for k_, v_ in popul.items():
            if k_ != 'samples':
                dout[ct][tr][k_] = deepcopy(v_)

        for sample in popul['samples']:
            sdict_ = {}
            for ks_, vs_ in sample.items():
                if ks_ in ['tc', 'old']:
                    continue

                elif ks_ == 'analysis':
                    sdict_[ks_] = {}
                    for ksa_, vsa_ in vs_.items():
                        if ksa_ == 'msd':
                            continue
                        sdict_[ks_][ksa_] = deepcopy(vsa_)

                else:
                    sdict_[ks_] = deepcopy(vs_)

            dout[ct][tr]['samples'].append(sdict_)

    if fpath.exists() and not rewrite:
        lg.warning(f"Analysis summary {fpath} already exists; skipping ...")

    else:
        with open(fpath, 'wb') as fh:
            pickle.dump(dout, fh)


def get_data_ccdf_from_fitobj(
        cells,
        celltype,
        treatment,
        data_type,
):
    if treatment in cells[celltype]:
        return (cells[celltype][treatment]['fits']
                [data_type]['fitobj'].ccdf())
