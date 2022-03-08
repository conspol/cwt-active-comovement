import re
from pathlib import Path

import chardet
import dill as pickle
import numpy as np
import pandas as pd
from loguru import logger as lg
from numba import jit
from numba.np.extensions import cross2d
from scipy.io import loadmat
from tqdm import tqdm

from constants import *


lg.add("execution_log.log", rotation="100MB", encoding="utf-8")


@jit(cache=True)
def dists_jit(
        lx,
        ly,
        nx,
        ny,
        max_start,
        min_stop,
        lt,
        nt,
):
    lx2 = lx[max_start - lt:min_stop - lt]
    ly2 = ly[max_start - lt:min_stop - lt]
    nx2 = nx[max_start - nt:min_stop - nt]
    ny2 = ny[max_start - nt:min_stop - nt]
    dists = np.sqrt(
        (lx2 - nx2) * (lx2 - nx2) + (ly2 - ny2) * (ly2 - ny2)
    )
    return dists


@jit(cache=True)
def corr_jit(
        lwav,
        nwav,
):
    return np.corrcoef(
        lwav.ravel(),
        nwav.ravel(),
    )[1, 0]


@jit(cache=True, nogil=True)
def window_minmax_jit(x):
    return (np.max(x) - np.min(x)) > 0.3


@jit(cache=True)
def haar_cwt_jit(
        x: np.ndarray,
        scale,
):
    n = int(scale / 2)
    l = len(x)
    coeffs = np.zeros(l)
    x = x - np.mean(x)

    for i in range(l):
        ip_ = min([i + n, l - 1])
        in_ = max([i - n, 0])
        coeffs[i] = (x[i:ip_].sum() - x[in_:i].sum()) / np.sqrt(scale)

    return coeffs


@jit(cache=True)
def cwt_jit(
        track: np.ndarray,
        minscale=1,
        maxscale=50,
):
    out = np.empty((maxscale, len(track)))

    for scale in range(minscale, maxscale + 1):
        out[scale-1] = haar_cwt_jit(track, scale)

    return out


def get_wav_overlap(
        wav,
        overlap_start,
        overlap_stop,
        tstart,
):
    """
    Indexing helper to get wavelet transform result range
    that overlaps with another one one.
    """
    return wav[
           :,
           overlap_start - tstart:overlap_stop - tstart
           ]


class TrackedCell:
    def __init__(
            self,
            name,
            filepath=None,
            df=None,
            lys_wav_path=None,
            np_wav_path=None,
            corr_thr=0.7,
            dist_thr=1,
            has_nps=True,
            minscale=1,
            maxscale=50,
            find_processed=True,
    ):
        self.name = name
        lg.info(f"Processing cell named {self.name}")
        self.cor_dists_file = Path('out').joinpath(self.name + '.pkl')
        self.cor_dists_file.parent.mkdir(exist_ok=True, parents=False)
        self.find_processed = find_processed

        self.lys_wav_path = lys_wav_path
        self.np_wav_path = np_wav_path
        self.wav_lys = None
        self.wav_np = None

        self.active_thr = None
        self.active_run_ixs = None
        self.active_lys_isx = None
        self.active_run_l = None
        self.active_run_t = None
        self.active_flight_l = None
        self.active_flight_t = None
        self.active_flight_se = None
        self.active_flight_i = None

        if filepath is None:
            self.df = df
        else:
            with open(filepath, 'rb') as f:
                fchar = chardet.detect(f.readline())
                lg.debug(f"Found file: \n[{filepath}]")
                lg.debug(f"Encoding is {fchar['encoding']}")

            lg.debug(f"Loading tracks into pd.DataFrame ...")
            self.df = pd.read_csv(filepath, encoding=fchar['encoding'])

        lg.debug(f"Columns: \n {self.df.columns}")
        lg.debug(f"DataFrame preview: \n {self.df}")

        # Clean up column names.
        clmns_ = [c_.strip().lower().replace('.', '_').replace(' ', '_')
                  for c_ in self.df.columns]
        clmns_ = [c_.replace('position', 'pos') for c_ in clmns_]
        clmns_ = [c_.replace('pos_', 'pos') for c_ in clmns_]
        self.df.columns = [re.sub(r'_\[.+', '', ll_) for ll_ in clmns_]

        if 'track_id' in self.df.columns:
            self.df.loc[:, 'id'] = self.df.loc[:, 'track_id']
            self.df = self.df.loc[self.df['id'].notna()].sort_values(
                by=['id', 'frame']).reset_index(drop=True)
            self.df.loc[:, 'id'] = self.df.loc[:, 'id'].astype(int)

        lg.debug(f"Column names were cleaned up: \n {self.df.columns}")

        self.wvscale = list(range(minscale, maxscale + 1))
        self.min_wv_scale = minscale
        self.max_wv_scale = maxscale

        self.cor = {
            'x': [],
            'y': [],
        }

        self.has_nps = has_nps

        self.nlys = 0
        self.nnp = 0

        self.df_np = None
        self.df_lys = None
        self.meandist = None
        self.cor = None
        self.lys_np_toverlap = None

        self.dlys = None
        self.dnp = None

        self.imatch = None
        self.match = None

        self.highcorx = None
        self.highcory = None
        self.highcor = None
        self.lowdist = None

        if self.cor_dists_file.is_file() and find_processed:
            lg.debug(f"Found file with calculated params of trajectories;\n"
                     f"loading [{self.cor_dists_file}]")
            self.load_pairwise_params_file()

        else:
            self.has_nps = self.make_split_df(has_nps=self.has_nps)

            if self.has_nps:
                self.correlate_wavelets()
                self.get_matching_pairs(corr_thr, dist_thr)

            self.save_pairwise_params_file()

    def __repr__(self):
        return f'TrackedCell(name={self.name})'

    def make_split_df(
            self,
            has_nps=True,
    ):
        """
        Splits DataFrame into lysosome and NP tracks.
        """

        # Make time indexes from timestamps.
        if 'time' not in self.df.columns:
            grp_ = self.df.groupby('id')
            timecols = []

            for idx, g_ in grp_:
                timecols.append(pd.Series(
                    pd.date_range(start='1900-01-01', periods=len(g_), freq='5S')))

            self.df.loc[self.df.index, 'time'] = pd.concat(timecols).values

        utime = self.df.time.unique()
        dtimemap = {
            tt_: it_ for it_, tt_ in zip(
                list(range(len(utime))),
                utime
            )
        }
        self.df['itime'] = self.df.time.map(dtimemap)

        if has_nps:
            id1 = self.df.loc[self.df.id == 1].index.values

            i_np_id_start = np.where(np.diff(id1) > 1)
            if len(i_np_id_start) != 1 or len(i_np_id_start[0]) != 1:
                lg.warning(f"There should be exactly one start point "
                           f"for NP objects. Check object IDs.")

            if i_np_id_start[0].shape == (0,):
                lg.warning(f"There were no tracks found for NPs. Processing only one track type.")
                has_nps = False

        if has_nps:
            np_id_start = id1[i_np_id_start[0][0] + 1]

            tid_lys = self.df.id.iloc[:np_id_start]
            tid_np = self.df.id.iloc[np_id_start:]

            self.nlys = tid_lys.max()
            self.nnp = tid_np.max()
            self.df_lys = self.df.iloc[:np_id_start]
            self.df_np = self.df.iloc[np_id_start:]

            cols2keep = [
                'id', 'itime', 'posx', 'posy', 'name', 'time',
            ]

            # if 'time' not in self.df_lys.columns:
            #     for df_ in (self.df_lys, self.df_np):
            #         grp_ = df_.groupby('id')
            #         timecols = []
            #
            #         for idx, g_ in grp_:
            #             timecols.append(pd.Series(
            #                 pd.date_range(start='1900-01-01', periods=len(g_), freq='1S')))
            #
            #         df_.loc[df_.index, 'time'] = pd.concat(timecols).values

            for col_ in cols2keep:
                if col_ not in self.df_lys.columns and col_ == 'name':
                    self.df_lys.loc[:, col_] = np.nan

            for col_ in cols2keep:
                if col_ not in self.df_np.columns and col_ == 'name':
                    self.df_np.loc[:, col_] = np.nan

            self.df_lys.loc[:, 'time'] = pd.to_datetime(
                self.df_lys.time,
                format='%M:%S.%f',
            )

            self.df_np.loc[:, 'time'] = pd.to_datetime(
                self.df_np.time,
                format='%M:%S.%f',
            )

            self.df_lys = self.df_lys.loc[:, cols2keep]
            self.df_np = self.df_np.loc[:, cols2keep]

            lg.info(f"There are {self.nlys} lysosome "
                    f"and {self.nnp} nanoparticle tracks; ")

            self.df_lys.set_index(['id', 'itime'], inplace=True)
            self.df_np.set_index(['id', 'itime'], inplace=True)

        if not has_nps:
            self.df_lys = self.df
            self.nlys = self.df_lys.id.max()

            cols2keep = [
                'id', 'itime', 'posx', 'posy', 'name', 'time',
            ]
            for col_ in cols2keep:
                if col_ not in self.df_lys.columns and col_ == 'name':
                    self.df_lys.loc[:, col_] = np.nan

            self.df_lys = self.df_lys.loc[:, cols2keep]

            lg.info(f"There are {self.nlys} lysosome tracks;")

            self.df_lys.loc[:, 'time'] = pd.to_datetime(
                self.df_lys.time,
                format='%M:%S.%f',
            )

            self.df_lys.set_index(['id', 'itime'], inplace=True)

        return has_nps

    def get_wavelets(
            self,
            lys_wav_path=None,
            np_wav_path=None,
    ):
        if lys_wav_path is None or np_wav_path is None:
            lg.debug(f"Calculating CWT ...")
            self.wav_lys = {'x': [], 'y': []}
            self.wav_np = {'x': [], 'y': []}
            minscale = self.min_wv_scale
            maxscale = self.max_wv_scale

            for id_, g_ in self.df_lys.groupby('id'):
                self.wav_lys['x'].append(cwt_jit(g_.posx.values, minscale, maxscale))
                self.wav_lys['y'].append(cwt_jit(g_.posy.values, minscale, maxscale))

            for id_, g_ in self.df_np.groupby('id'):
                self.wav_np['x'].append(cwt_jit(g_.posx.values, minscale, maxscale))
                self.wav_np['y'].append(cwt_jit(g_.posy.values, minscale, maxscale))

        else:
            lg.debug(f"Loading wavelets for lysosome tracks "
                     f"from: \n [{lys_wav_path}] ...")
            ll = loadmat(lys_wav_path)
            lg.debug(f"Loading wavelets for nanoparticle tracks "
                     f"from: \n [{np_wav_path}] ...")
            nn = loadmat(np_wav_path)

            self.wav_lys = {
                'x': ll['lwavx'][:, 0],
                'y': ll['lwavy'][:, 0],
            }

            self.wav_np = {
                'x': nn['nwavx'][:, 0],
                'y': nn['nwavy'][:, 0],
            }

    def correlate_wavelets(
            self,
            t_threshold=20,
    ):
        grplys = self.df_lys.reset_index(level=1).groupby('id')
        grpnp = self.df_np.reset_index(level=1).groupby('id')

        # Filtering out the short tracks
        self.df_lys = grplys.filter(lambda x: len(x) > t_threshold).set_index('itime', append=True)
        self.df_np = grpnp.filter(lambda x: len(x) > t_threshold).set_index('itime', append=True)

        grplys = self.df_lys.reset_index(level=1).groupby('id')
        grpnp = self.df_np.reset_index(level=1).groupby('id')

        self.dlys = {
            'tmin': grplys.itime.min(),
            'tmax': grplys.itime.max(),
        }

        self.dnp = {
            'tmin': grpnp.itime.min(),
            'tmax': grpnp.itime.max(),
        }

        grplys = self.df_lys.groupby('id')
        grpnp = self.df_np.groupby('id')

        mtx_shape = (self.nlys, self.nnp)
        overlaps = np.zeros(mtx_shape, dtype=np.bool_)
        meandist = np.zeros(mtx_shape)
        cor = {
            'x': np.zeros(mtx_shape),
            'y': np.zeros(mtx_shape),
        }

        if self.cor_dists_file.is_file() and self.find_processed:
            lg.debug(f"Found file with calculated pairwise params of trajectories;\n"
                     f"loading [{self.cor_dists_file}]")
            with open(self.cor_dists_file, 'rb') as f:
                precalc = pickle.load(f)
                self.meandist = precalc['meandist']
                self.cor = precalc['cor']
                self.lys_np_toverlap = precalc['overlaps']

        else:
            self.get_wavelets(
                lys_wav_path=self.lys_wav_path,
                np_wav_path=self.np_wav_path,
            )

            lg.debug(f"Splitting NPs by track ID into dict ...")
            dgrp_np = {}
            for grpi_, grp_ in grpnp:
                dgrp_np[grpi_] = grp_

            lg.debug(f"Calculating pairwise params of trajectories ...")
            for li_ in tqdm(range(self.nlys)):
                lid = li_ + 1
                try:
                    lysg = grplys.get_group(lid)
                except KeyError:
                    lg.warning(f"No lysosome track with ID {lid} found.")
                    continue

                if len(lysg) != self.wav_lys['x'][li_].shape[1] \
                        or len(lysg) != self.wav_lys['y'][li_].shape[1]:
                    lg.warning(f"Length of lysosome track {lid} "
                               f"does not match the wavelet length; skipping.")

                lysg_posx = lysg.posx.values
                lysg_posy = lysg.posy.values

                for ni_ in range(self.nnp):
                    nid = ni_ + 1
                    npg = dgrp_np[nid]

                    if len(npg) != self.wav_np['x'][ni_].shape[1] \
                            or len(npg) != self.wav_np['y'][ni_].shape[1]:
                        lg.warning(f"Length of nanoparticle track {ni_} "
                                   f"does not match the wavelet length; skipping.")
                        continue

                    min_stop = min(
                        (self.dlys['tmax'][lid], self.dnp['tmax'][nid]))
                    max_start = max(
                        (self.dlys['tmin'][lid], self.dnp['tmin'][nid]))
                    overlaps[li_, ni_] = max_start < min_stop

                    if overlaps[li_, ni_]:
                        lwav = {}
                        nwav = {}
                        tstart_lys = self.dlys['tmin'][lid]
                        tstart_np = self.dnp['tmin'][nid]

                        for ax_ in ['x', 'y']:
                            lwav[ax_] = get_wav_overlap(
                                self.wav_lys[ax_][li_],
                                max_start,
                                min_stop,
                                tstart_lys,
                            )
                            nwav[ax_] = get_wav_overlap(
                                self.wav_np[ax_][ni_],
                                max_start,
                                min_stop,
                                tstart_np,
                            )

                            cor[ax_][li_, ni_] = corr_jit(lwav[ax_], nwav[ax_])

                        meandist[li_, ni_] = dists_jit(
                            lysg_posx,
                            lysg_posy,
                            npg.posx.values,
                            npg.posy.values,
                            max_start,
                            min_stop,
                            tstart_lys,
                            tstart_np,
                        ).mean()

                    else:
                        meandist[li_, ni_] = np.nan
                        cor['x'][li_, ni_] = np.nan
                        cor['y'][li_, ni_] = np.nan

            lg.debug(f"Pairwise parameters calculated, saving to file:\n"
                     f"[{self.cor_dists_file}]")
            self.cor = cor
            self.meandist = meandist
            self.lys_np_toverlap = overlaps

    def save_pairwise_params_file(self):
        with open(self.cor_dists_file, 'wb') as f:
            pickle.dump(
                {
                    'meandist': self.meandist,
                    'cor': self.cor,
                    'df_lys': self.df_lys,
                    'df_np': self.df_np,
                    'overlaps': self.lys_np_toverlap,
                    'imatch': self.imatch,
                },
                f,
            )

    def load_pairwise_params_file(self):
        with open(self.cor_dists_file, 'rb') as f:
            precalc = pickle.load(f)
            self.meandist = precalc['meandist']
            self.cor = precalc['cor']
            self.df_lys = precalc['df_lys']
            self.df_np = precalc['df_np']
            self.lys_np_toverlap = precalc['overlaps']
            self.imatch = precalc['imatch']

    def get_matching_pairs(
            self,
            corr_thr,
            dist_thr,
    ):
        self.highcorx = self.cor['x'] > corr_thr
        self.highcory = self.cor['y'] > corr_thr
        self.highcor = self.highcory & self.highcorx
        self.lowdist = self.meandist < dist_thr

        self.match = self.highcor & self.lowdist
        self.imatch = np.where(self.match)
        lg.info(f"Found {len(self.imatch[0])} matching "
                f"lysosome-NPs pairs, with {len(np.unique(self.imatch[0]))} "
                f"unique lysosomes.")

    def get_active_runs(
            self,
            wv_scale=WV_SCALE,
            prefactor=0.75,
            wvpad=WV_PAD,
    ):
        lg.debug(f"{self.name} | Calculating active runs ...")
        if self.wav_lys is None or self.wav_np is None:
            lg.debug("No wavelet data found, loading from the files ...")
            self.get_wavelets(
                lys_wav_path=self.lys_wav_path,
                np_wav_path=self.np_wav_path,
            )

        self.active_lys_isx = []
        self.active_run_ixs = {}
        self.active_run_l = {}
        self.active_run_t = {}

        nu = np.sqrt(wv_scale / 2)

        run_ixs = {}

        # Filter out the most "lazy" lysosomes
        # that clearly are not active (moving less than 0.3 um in total)
        # using the rolling window function.
        lfilt = []
        for i_, id_ in enumerate(
                self.df_lys.index.get_level_values(0).unique()):

            df_ = self.df_lys.loc[id_, ('posx', 'posy')]
            rolw_ = df_.rolling(10).apply(window_minmax_jit, engine='numba', raw=True)

            if rolw_.any(axis=None):
                lfilt.append(i_)

        for ax_ in ('x', 'y'):
            for i_ in lfilt:
                w_ = self.wav_lys[ax_][i_]

                if w_.shape[1] > 20:
                    # Calculating "universal threshold" using
                    # wavelet scale 2
                    s_ = w_[wv_scale, wvpad:-wvpad]

                    # Using only non-zero wavelet coefficients
                    nz2 = np.nonzero(w_[2])
                    s2 = np.median(np.abs(w_[2, nz2])) / 0.6745

                    # Prefactor is <1 since we excluded zeroes,
                    # hence "artificially" (due to the discreteness
                    # of data) raised the median value.
                    # The appropriate prefactor should be chosen
                    # based on the data.
                    unithr = prefactor * nu * s2 * \
                             np.sqrt(2 * np.log(s_.shape[0]))

                    run_ = np.where(np.abs(s_) > unithr)[0] + wvpad

                    if len(run_) > 2:
                        if i_ not in run_ixs:
                            run_ixs[i_] = {}

                        run_ixs[i_][ax_] = run_

        for i_, run_ in run_ixs.items():

            # Unifying runs for both axes
            if 'x' in run_:
                lx = list(run_['x'])
            else:
                lx = []
            if 'y' in run_:
                ly = list(run_['y'])
            else:
                ly = []

            run2d = np.array(sorted(set(lx + ly)))

            self.active_lys_isx.append(i_)

            runstart = run2d[np.where(np.diff(run2d) > 1)[0] + 1]
            runstart = np.hstack((run2d[0], runstart))
            runend = run2d[np.where(np.diff(run2d) > 1)[0]]
            runend = np.hstack((runend, run2d[-1]))

            run_t = runend - runstart + 1
            short = run_t < 3
            runstart = runstart[~short]
            runend = runend[~short]

            run_t = runend - runstart + 1
            run_l = []

            id_ = self.df_lys.index.get_level_values(0).unique()[i_]
            runs = self.df_lys.loc[id_, ('posx', 'posy')].diff()

            dists = np.linalg.norm(runs.iloc[1:], axis=1)

            # When the active run starts close to the end of the
            # trajectory in time (the unused region of wavelet),
            # sometimes it leads to zero-length active runs.
            # We remove such runs.
            zeroruns = []

            for si_, (ss_, se_) in enumerate(
                    zip(runstart, runend + 1)):

                run_l_ = dists[ss_:se_].sum()
                run_l.append(run_l_)
                if run_l_ < 0.2:
                    zeroruns.append(si_)

            if zeroruns:
                run_t = np.delete(run_t, zeroruns)
                run_l = np.delete(run_l, zeroruns)
                runstart = np.delete(runstart, zeroruns)
                runend = np.delete(runend, zeroruns)

            self.active_run_t[i_] = run_t
            self.active_run_l[i_] = np.array(run_l)
            self.active_run_ixs[i_] = (runstart, runend)

    def get_active_flights(
            self,
            threshold=0.4,
    ):
        """
        Getting active flights for each lysosome using error-radius method
        on previously detected runs.
        """

        # Instead of indices of runs, we will use the indices of the
        # active runs that constitute flight.
        # _se stands for "start-end" indices.
        self.active_flight_se = {}
        # Indices of runs that constitute flight.
        self.active_flight_i = {}
        self.active_flight_l = {}
        self.active_flight_t = {}

        lg.debug("Calculating active flights ...")

        for i_, ix_ in self.active_run_ixs.items():
            track = self.df_lys.loc[i_ + 1, ('posx', 'posy')].values

            if len(ix_[0]) > 1:
                self.active_flight_t[i_] = []
                self.active_flight_l[i_] = []
                lflight, lflight_i = runs_into_flights(track, ix_, threshold)
                self.active_flight_se[i_] = lflight
                self.active_flight_i[i_] = lflight_i
                for fi_ in lflight_i:
                    self.active_flight_t[i_].append(self.active_run_t[i_][fi_].sum())
                    self.active_flight_l[i_].append(self.active_run_l[i_][fi_].sum())

        # Adding runs that are not part of any flight.
        for ri_, runs_ in self.active_run_ixs.items():
            if ri_ in self.active_flight_i and len(self.active_flight_i[ri_]) > 0:
                ts_ = self.active_run_t[ri_].copy()
                ls_ = self.active_run_l[ri_].copy()

                todel_i = []

                for fl_ in self.active_flight_i[ri_]:
                    todel_i.extend(fl_)

                ts_ = np.delete(ts_, todel_i)
                ls_ = np.delete(ls_, todel_i)

                self.active_flight_t[ri_].extend(ts_)
                self.active_flight_l[ri_].extend(ls_)

            else:
                self.active_flight_t[ri_] = self.active_run_t[ri_]
                self.active_flight_l[ri_] = self.active_run_l[ri_]


@jit(cache=True, nopython=True)
def _jit_get_dists(
        track,
        ibeg,
        iend,
):
    """
    Calculates the distance between the points and the start-end line.
    """
    run = track[ibeg:iend + 1]
    beg = run[0]
    end = run[-1]
    line = end - beg

    return cross2d(line, beg - run) / np.linalg.norm(line)


def runs_into_flights(
        track,
        se: np.ndarray,
        threshold,
):
    """
    Finds if runs belong to the same flight using the error-radius method.
    The resultsing angles distribution should be checked for autocorellation manually,
    and the threshold should be chosen based on that.

    :param track:       Track to process
    :param se:          Array of starts and ends of runs; shape (2, n).
    :param threshold:   Error radius threshold.
    :return:
    """

    thresh_ = threshold
    begs = se[0]
    ends = se[1]

    # Expand the threshold if the runs are wide.
    # Also, get angles between runs.
    langles = []
    for i_ in range(len(begs)):

        dists = _jit_get_dists(track, begs[i_], ends[i_])

        if np.any(dists > thresh_):
            thresh_ = np.max(dists) * 1.27

        if i_ > 0:
            v1 = track[ends[i_ - 1]] - track[begs[i_ - 1]]
            v2 = track[ends[i_]] - track[begs[i_]]
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            langles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    lflights = []
    lflights_i = []
    connected = False

    # Find flights
    for bix_ in range(len(begs) - 1):
        ibeg = se[0][bix_]

        # We need to get flights from multiple runs,
        # so starting a loop from the end of the next run.
        eix_ = bix_ + 1
        iend = se[1][eix_]

        dists = _jit_get_dists(track, ibeg, iend)

        if np.all(np.abs(dists) < thresh_) and \
                langles[eix_ - 1] < np.deg2rad(115):
            if connected:
                lflights[-1][1] = iend
                lflights_i[-1].extend([eix_])
            else:
                lflights.append([ibeg, iend])
                lflights_i.append([bix_, eix_])
                connected = True

        else:
            connected = False

    return lflights, lflights_i
