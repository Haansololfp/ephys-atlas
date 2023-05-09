# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from one.api import ONE
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se
from spike_psvae import subtract
from pathlib import Path
import subprocess
import fileinput

# %%
Benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724', 
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7', 
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e', 
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd', 
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1', 
                  '749cb2b7-e57e-4453-a794-f6230e4d0226', 
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c', 
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd', 
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0', 
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea', 
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c', 
                  'eebcaf65-7fa4-4118-869d-a084e84530e2', 
                  'fe380793-8035-414e-b000-09bfe5ece92a']

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets'
for i in range(1):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    rec = se.read_ibl_streaming_recording(
        eID,
        probe + '.ap',
        cache_folder="/local/sicache",
    )
    
    fs = int(rec.get_sampling_frequency())

    rec = rec.frame_slice(start_frame=int(2000*fs),end_frame=int(2180*fs))

    rec = si.highpass_filter(rec)
    rec = si.phase_shift(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
    print(f"{bad_channel_ids=}")
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    # we had been working with this before -- should switch to MAD,
    # but we need to rethink the thresholds
    rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100)
    print(rec)
    # try:
    out_dir = main_dir + '/' + eID + '_' + probe
    # # !mkdir {out_dir}
    sub_h5 = subtract.subtraction(
            rec,
            out_folder=out_dir,
            thresholds=[12, 10, 8, 6, 5],
            n_sec_pca=40,
            save_subtracted_tpca_projs=False,
            save_cleaned_tpca_projs=False,
            save_denoised_tpca_projs=False,
            n_jobs=14,
            loc_workers=1,
            overwrite=False,
            # n_sec_chunk=args.batchlen,
            save_cleaned_pca_projs_on_n_channels=5,
            loc_feature=("ptp", "peak"),
        )
        # sub_h5 = subtract.subtraction(
        #     rec,
        #     out_folder=out_dir,
        #     thresholds=[12, 10, 8, 6],#[12, 10, 8, 6, 5],
        #     n_sec_pca=40,
        #     save_subtracted_tpca_projs=False,
        #     save_cleaned_tpca_projs=False,
        #     # save_denoised_tpca_projs=True,
        #     save_denoised_waveforms=True,
        #     n_jobs=14,
        #     loc_workers=1,
        #     overwrite=False,
        # )
    # except:
    #     continue

# %% jupyter={"outputs_hidden": true}
from brainbox.io.spikeglx import Streamer
import sys
import subprocess
import spikeinterface.full as sf
webclient = one.alyx
cache_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe"
destripe_py_dir = '/moto/stats/users/hy2562/spike-psvae/scripts/destripe.py'
save_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract"

stream_type = "ap"
remove_cached = True
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)

    ibl_treamer = Streamer(pid=pID, one=one, typ=stream_type, cache_folder=cache_folder, remove_cached=remove_cached)
    ibl_treamer._download_raw_partial(first_chunk=2000, last_chunk=2179)
    cbin_url = ibl_treamer.url_cbin
    
    alyx_base_path = one.eid2path(eID)
    relpath= alyx_base_path.relative_to(one.cache_dir)
    
    cbin_parent_dir = cache_folder + '/' + str(relpath) + '/raw_ephys_data/' + probe + '/chunk_002000_to_002179'
    
    cbin_dir = list(Path(cbin_parent_dir).glob('*.cbin'))[0]
    meta_dir = list(Path(cbin_parent_dir).glob('*.meta'))[0]
    ch_dir = list(Path(cbin_parent_dir).glob('*.ch'))[0]
    
    for line in fileinput.input(meta_dir, inplace=1):
        if 'fileTimeSecs' in line:
            line = 'fileTimeSecs=180\n'
        sys.stdout.write(line)
    
    subprocess.run(["python",
                    destripe_py_dir,
                    str(cbin_dir),
                   ],
                   check = True
                  )
    
    destriped_cbin_dir = list(Path(cbin_parent_dir).glob('destriped_*.cbin'))[0]
    destriped_meta_dir = list(Path(cbin_parent_dir).glob('destriped_*.meta'))[0]
    
    # destriped_cbin_dir = cbin_parent_dir + destriped_cbin_dir.name
    # destriped_meta_dir = cbin_parent_dir + destriped_meta_dir.name
    
    
    out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID + '/'
    # !mkdir {out_dir}
    
    # !mv {str(destriped_cbin_dir)} {out_dir}
    # !mv {str(destriped_meta_dir)} {out_dir}
    
    
    save_destriped_ch_dir = out_dir + 'destriped_' + ch_dir.name
     
    # !cp {ch_dir} {save_destriped_ch_dir}
    
    out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name
    
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    try:
        sub_h5 = subtract.subtraction(
                        rec,
                        out_folder=out_dir,
                        thresholds=[12, 10, 8, 6, 5],
                        n_sec_pca=40,
                        save_subtracted_tpca_projs=False,
                        save_cleaned_tpca_projs=False,
                        save_denoised_tpca_projs=False,
                        save_denoised_waveforms=True,
                        n_jobs=14,
                        loc_workers=1,
                        overwrite=False,
                        # n_sec_chunk=args.batchlen,
                        save_cleaned_pca_projs_on_n_channels=5,
                        loc_feature=("ptp", "peak"),
                    )
    except:
        continue
    

# %%
i

# %%
list(Path(cbin_parent_dir).glob('*.cbin'))

# %% jupyter={"outputs_hidden": true}
destriped_cbin_dir = list(Path(cbin_parent_dir).glob('destriped_*.cbin'))[0]
destriped_meta_dir = list(Path(cbin_parent_dir).glob('destriped_*.meta'))[0]

# destriped_cbin_dir = cbin_parent_dir + destriped_cbin_dir.name
# destriped_meta_dir = cbin_parent_dir + destriped_meta_dir.name


out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID + '/'
# !mkdir {out_dir}

# !mv {str(destriped_cbin_dir)} {out_dir}
# !mv {str(destriped_meta_dir)} {out_dir}


save_destriped_ch_dir = out_dir + 'destriped_' + ch_dir.name

# !cp {ch_dir} {save_destriped_ch_dir}

out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
rec_cbin = sf.read_cbin_ibl(Path(out_dir))

destriped_cbin = out_dir + '/' +destriped_cbin_dir.name

rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

sub_h5 = subtract.subtraction(
                rec,
                out_folder=out_dir,
                thresholds=[12, 10, 8, 6, 5],
                n_sec_pca=40,
                save_subtracted_tpca_projs=False,
                save_cleaned_tpca_projs=False,
                save_denoised_tpca_projs=False,
                save_denoised_waveforms=True,
                n_jobs=14,
                loc_workers=1,
                overwrite=False,
                # n_sec_chunk=args.batchlen,
                save_cleaned_pca_projs_on_n_channels=5,
                loc_feature=("ptp", "peak"),
            )


# %%
import fileinput

for line in fileinput.input(meta_dir, inplace=1):
    if 'fileTimeSecs' in line:
        line = 'fileTimeSecs = 180\n'
    sys.stdout.write(line)

# %%
# !ls /moto/stats/users/hy2562/projects/ephys_atlas/improved_destripecortexlab/Subjects/KS046/2020-12-03/001/raw_ephys_data/probe00/chunk_002000_to_002179

# %%
relpath= alyx_base_path.relative_to(one.cache_dir)

# %%
i=0
pID = Benchmark_pids[i]
eID, probe = one.pid2eid(pID)
alyx_base_path = one.eid2path(eID)

# %% jupyter={"outputs_hidden": true}
one.list_datasets(eID)

# %%
rel_path = Path(rel_path[0])

# %%
rel_path.parent

# %%
alyx_base_path.relative_to(one.cache_dir)

# %%
cbin_path = alyx_base_path.relative_to(one.cache_dir)/rel_path.parent

# %%
cbin_path

# %%
alyx_base_path = one.eid2path(eID)

# %%
searchdir = alyx_base_path.relative_to(one.cache_dir)/rel_path.parent
pattern = Path(rel_path.name).with_suffix(f".*.cbin")
glob = list(searchdir.glob(str(pattern)))

# %%
from brainbox.io.spikeglx import Streamer
eID, probe = one.pid2eid(pID)
stream_type = 'ap'
cache_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe"#"/local/sicache"
remove_cached = False
ibl_treamer = Streamer(pid=pID, one=one, typ=stream_type, cache_folder=cache_folder, remove_cached=remove_cached)
ibl_treamer._download_raw_partial(first_chunk=2000, last_chunk=2180)


# %%
cbin_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/cortexlab/Subjects/KS046/2020-12-03/001/raw_ephys_data/probe00/chunk_002000_to_002180/_spikeglx_ephysData_g0_t0.imec0.ap.stream.cbin'

# %%
import subprocess

# %%
# %load_ext autoreload
# %autoreload 2

# %% jupyter={"outputs_hidden": true}
destripe_dir = '/moto/stats/users/hy2562/spike-psvae/scripts/destripe.py'
subprocess.run(["python",
                destripe_dir,
                cbin_dir,
               ],
              )

# %%
cbin_url = ibl_treamer.url_cbin

# %%
cbin_pth = Path(cbin_url)

# %%
cbin_url

# %%
cbin_pth.parent

# %%
ch_path = list(cbin_path.parent.glob("*ap*.ch"))
assert len(ch_path) == 1
ch_path = ch_path[0]
meta_path = list(cbin_path.parent.glob("*ap*.meta"))
assert len(meta_path) == 1
meta_path = meta_path[0]

# %%
list(Path(destriped_cbin).parent.glob("*.ch"))

# %%
import spikeinterface.full as si
destriped_cbin = Path('/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/cortexlab/Subjects/KS046/2020-12-03/001/raw_ephys_data/probe00/chunk_002000_to_002180/destriped/destriped__spikeglx_ephysData_g0_t0.imec0.ap.stream.cbin')
rec_cbin = si.read_cbin_ibl(destriped_cbin.parent)
rec = si.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

# %%
subcache = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/cortexlab/Subjects/KS046/2020-12-03/001/raw_ephys_data/probe00/chunk_002000_to_002180/destriped'
sub_h5 = subtract.subtraction(
                rec,
                out_folder=subcache,
                thresholds=[12, 10, 8, 6, 5],
                n_sec_pca=40,
                save_subtracted_tpca_projs=False,
                save_cleaned_tpca_projs=False,
                save_denoised_tpca_projs=False,
                n_jobs=14,
                loc_workers=1,
                overwrite=False,
                # n_sec_chunk=args.batchlen,
                save_cleaned_pca_projs_on_n_channels=5,
                loc_feature=("ptp", "peak"),
            )

# %%
import h5py
sub_h51 = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/cortexlab/Subjects/KS046/2020-12-03/001/raw_ephys_data/probe00/chunk_002000_to_002180/destriped/subtraction.h5'
with h5py.File(sub_h51) as h5:
    z_abs1 = h5["localizations"][:, 2]
    x1 = h5["localizations"][:, 0]
    y1 = h5["localizations"][:, 1]
    times1 = h5["spike_index"][:, 0] / 30_000
    maxptps1 = h5["maxptps"][:]



# %%
eID

# %%
sub_h52 = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets/69c9a415-f7fa-4208-887b-1417c1479b48_probe00/subtraction.h5'
with h5py.File(sub_h52) as h5:
    z_abs2 = h5["localizations"][:, 2]
    x2 = h5["localizations"][:, 0]
    y2 = h5["localizations"][:, 1]
    times2 = h5["spike_index"][:, 0] / 30_000
    maxptps2 = h5["maxptps"][:]



# %%
import numpy as np
np.max(maxptps2)

# %%
np.min(maxptps2)

# %%
np.max(maxptps1)

# %%
np.min(maxptps1)

# %%
len(maxptps1)

# %%
import matplotlib as mpl
which = maxptps1>8
plt.figure(figsize = [2, 20])
cmps = np.clip(maxptps1[which], 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
plt.scatter(x1[which], z_abs1[which], c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])

# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize = [20, 20])
axs[0].scatter(times1, z_abs1, c = maxptps1, s = 0.1)
axs[0].set_xlim([0, 180])
axs[1].scatter(times2, z_abs2, c = maxptps2, s = 0.1)
axs[1].set_xlim([0, 180])

# %%
import numpy as np
from one.api import ONE, OneAlyx
import spikeinterface.full as si
import spikeinterface.extractors as se
from spike_psvae import subtract
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import shutil
import argparse
import subprocess

# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("minix", type=int)
    ap.add_argument("maxix", type=int, default=None)
    ap.add_argument("--njobs", type=int, default=20)
    ap.add_argument("--nspca", type=int, default=40)
    ap.add_argument("--batchlen", type=float, default=1)
    ap.add_argument("--locworkers", type=int, default=2)

    args = ap.parse_args()
    
    import torch
    print(f"{torch.cuda.is_available()=}")

    # %%
    # OneAlyx.setup(base_url='https://alyx.internationalbrainlab.org', make_default=True)

    # %%
    one = ONE()

    # %%
    sessions_rep_site = one.alyx.rest('sessions', 'list', dataset_types='spikes.times', tag='2022_Q2_IBL_et_al_RepeatedSite')
    sessions_rep_site = list(sorted(sessions_rep_site, key=lambda session: session['id']))
    
    minix = args.minix
    maxix = args.maxix
    if maxix is None:
        maxix = len(sessions_rep_site)
    print(f"{minix=} {maxix=}")

    # %%
    sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")

    def eid2sdscpath(eid):
        pids, probes = one.eid2pid(eid)
        alyx_base_path = one.eid2path(eid)
        paths = {}
        for pid, probe in zip(pids, probes):
            rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
            assert len(rel_path) == 1
            rel_path = Path(rel_path[0])
            searchdir = sdsc_base_path / alyx_base_path.relative_to(one.cache_dir) / rel_path.parent
            pattern = Path(rel_path.name).with_suffix(f".*.cbin")
            glob = list(searchdir.glob(str(pattern)))
            assert len(glob) == 1
            paths[probe] = pid, glob[0]
            assert paths[probe][1].exists()
        return paths


    # %%

    outdir = Path("/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe")
    outdir.mkdir(exist_ok=True, parents=True)
    subcache = Path("/local/sicache")
    dscache = Path("/tmp/dscache")
    dscache.mkdir(exist_ok=True)

    # %%

    # -----------------------------------------
    for session in sessions_rep_site[minix:maxix]:

        paths = eid2sdscpath(session['id'])
        if "probe00" in paths:
            pid, cbin_path = paths["probe00"]
        else:
            print("No probe00, skipping for now")
            
        sessdir = outdir / f"pid{pid}"
        sessdir.mkdir(exist_ok=True)
        
        metadata = dict(
            probe="probe00",
            pid=pid,
            cbin_path=cbin_path,
            session=session,
        )
        
        if (sessdir / "metadata.pkl").exists():
            with open(sessdir / "metadata.pkl", "rb") as sess_jar:
                meta = pickle.load(sess_jar)
                if "done" in meta and meta["done"]:
                    print(session['id'], "already done.")
                    assert (sessdir / "subtraction.h5").exists()

                    # -----------------------------------------
                    continue

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            pickle.dump(metadata, sess_jar)

        ch_path = list(cbin_path.parent.glob("*ap*.ch"))
        assert len(ch_path) == 1
        ch_path = ch_path[0]
        meta_path = list(cbin_path.parent.glob("*ap*.meta"))
        assert len(meta_path) == 1
        meta_path = meta_path[0]

        print("-" * 50)
        print(session['id'], cbin_path)
        rec_cbin = si.read_cbin_ibl(str(cbin_path.parent), cbin_file=str(cbin_path), ch_file=str(ch_path), meta_file=str(meta_path))
        print(rec_cbin)
        fs = int(rec_cbin.get_sampling_frequency())
        
        # copy to temp dir
        dscache.mkdir(exist_ok=True)
        cbin_rel = cbin_path.stem.split(".")[0] + ("".join(s for s in cbin_path.suffixes if len(s) < 10))
        shutil.copyfile(cbin_path, dscache / cbin_rel)
        meta_rel = meta_path.stem.split(".")[0] + ("".join(s for s in meta_path.suffixes if len(s) < 10))
        shutil.copyfile(meta_path, dscache / meta_rel)
        ch_rel = ch_path.stem.split(".")[0] + ("".join(s for s in ch_path.suffixes if len(s) < 10))
        shutil.copyfile(ch_path, dscache / ch_rel)
        
        # run destriping
        destriped_cbin = dscache / f"destriped_{cbin_rel}"
        if not destriped_cbin.exists():
            try:
                subprocess.run(
                    [
                        "python",
                        str(Path(__file__).parent / "destripe.py"),
                        str(dscache / cbin_rel),
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                    metadata['destriping_error'] = e
                    pickle.dump(metadata, sess_jar)
                if (dscache / f"destriped_{cbin_rel}").exists():
                    (dscache / f"destriped_{cbin_rel}").unlink()
                continue
        
        assert destriped_cbin.exists()
        
        rec = si.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
        rec.set_probe(rec_cbin.get_probe(), in_place=True)
        fs = rec.get_sampling_frequency()

        ttt = rec.get_traces(start_frame=rec.get_num_samples() // 2, end_frame=rec.get_num_samples() // 2+1000)
        print(f"{ttt.min()=} {ttt.max()=}")

        # if subcache.exists():
        #     shutil.rmtree(subcache)

        try:
            sub_h5 = subtract.subtraction(
                rec,
                out_folder=subcache,
                thresholds=[12, 10, 8, 6, 5],
                n_sec_pca=args.nspca,
                save_subtracted_tpca_projs=False,
                save_cleaned_tpca_projs=False,
                save_denoised_tpca_projs=False,
                n_jobs=args.njobs,
                loc_workers=args.locworkers,
                overwrite=False,
                n_sec_chunk=args.batchlen,
                save_cleaned_pca_projs_on_n_channels=5,
                loc_feature=("ptp", "peak"),
            )
            shutil.copy(sub_h5, sessdir)
        except Exception as e:
            with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                metadata['subtraction_error'] = e
                pickle.dump(metadata, sess_jar)
        finally:
            shutil.rmtree(subcache)
            shutil.rmtree(dscache)

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            metadata['done'] = True
            pickle.dump(metadata, sess_jar)

# %%
