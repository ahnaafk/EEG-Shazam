__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import os
import datetime

import numpy as np

import mne
from mne.io.edf.edf import read_raw_edf

from scipy.io import loadmat

from constants import STIMULUS_IDS
from metadata import load_stimuli_metadata, load_stimuli_metadata_map


def get_event_id(stimulus_id, condition):
    return stimulus_id * 10 + condition

def decode_event_id(event_id):
    if event_id < 1000:
        stimulus_id = event_id / 10
        condition = event_id % 10
        return stimulus_id, condition
    else:
        return event_id

def default_beat_event_id_generator(stimulus_id, condition, cue, beat_count):
    if cue:
        cue = 0
    else:
        cue = 10
    return 100000 + stimulus_id * 1000 + condition * 100 + cue + beat_count

def simple_beat_event_id_generator(stimulus_id, condition, cue, beat_count):
    return 10000

def generate_beat_events(trial_events,                  # base events as stored in raw fif files
                         include_cue_beats=True,        # generate events for cue beats as well?
                         use_audio_onset=True,          # use the more precise audio onset marker (code 1000) if present
                         exclude_stimulus_ids=[],
                         exclude_condition_ids=[],
                         beat_event_id_generator=default_beat_event_id_generator,
                         sr=512.0,                      # sample rate, correct value important to compute event frames
                         verbose=False,
                         version=None):

    ## prepare return value
    beat_events = []

    ## get stimuli meta information
    meta = load_stimuli_metadata_map(version=version)
    beats = load_stimuli_metadata_map('beats', verbose=verbose, version=version)

    if include_cue_beats:
        cue_beats = load_stimuli_metadata_map('cue_beats')

        ## determine the number of cue beats
        num_cue_beats = dict()
        for stimulus_id in STIMULUS_IDS:
            num_cue_beats[stimulus_id] = \
                meta[stimulus_id]['beats_per_bar'] * meta[stimulus_id]['cue_bars']
        if verbose:
            print (num_cue_beats)


    ## helper function to add a single beat event
    def add_beat_event(etime, stimulus_id, condition, beat_count, cue=False):
        etype = beat_event_id_generator(stimulus_id, condition, cue, beat_count)
        beat_events.append([etime, 0, etype])
        if verbose:
            print(beat_events[-1])

    ## helper function to add a batch of beat events
    def add_beat_events(etimes, stimulus_id, condition, cue=False):
        beats_per_bar = meta[stimulus_id]['beats_per_bar']

        for i, etime in enumerate(etimes):
            beat_count = (i % beats_per_bar) + 1
            add_beat_event(etime, stimulus_id, condition, beat_count, cue)

    for i, event in enumerate(trial_events):
        etype = event[2]
        etime = event[0]

        if verbose:
            print('{:4d} at {:8d}'.format(etype, etime))

        if etype >= 1000: # stimulus_id + condition
            continue

        stimulus_id, condition = decode_event_id(etype)

        if stimulus_id in exclude_stimulus_ids or condition in exclude_condition_ids:
            continue  # skip excluded

        trial_start = etime # default: use trial onset
        if use_audio_onset and condition < 3:
            # Note: conditions 3 and 4 have no audio cues
            next_event = trial_events[i+1]
            if next_event[2] == 1000: # only use if audio onset
                trial_start = next_event[0]

        if verbose:
            print('Trial start at {}'.format(trial_start))

        if condition < 3: # cued
            print("This is the length of the cue: ", meta[stimulus_id]["length_with_cue"])
            offset = sr * meta[stimulus_id]['length_of_cue']

            if include_cue_beats:
                cue_beat_times = trial_start + np.floor(sr * cue_beats[stimulus_id])
                cue_beat_times = cue_beat_times[:num_cue_beats[stimulus_id]]  # truncate at num_cue_beats
                cue_beat_times = np.asarray(cue_beat_times, dtype=int)
                if verbose:
                    print(cue_beat_times)
                add_beat_events(cue_beat_times, stimulus_id, condition, cue=True)
        else:
            offset = 0 # no cue

        beat_times = trial_start + offset + np.floor(sr * beats[stimulus_id])
        beat_times = np.asarray(beat_times, dtype=int)
        if verbose:
            print(beat_times[:5], '...')
        add_beat_events(beat_times, stimulus_id, condition)

    beat_events = np.asarray(beat_events, dtype=int)

    return beat_events


def merge_trial_and_audio_onsets(raw, use_audio_onsets=True, inplace=True, stim_channel='STI 014', verbose=None):
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)

    merged = list()
    last_trial_event = None
    for i, event in enumerate(events):
        etype = event[2]
        if etype < 1000 or etype == 1111: # trial or noise onset
            if use_audio_onsets and events[i+1][2] == 1000: # followed by audio onset
                onset = events[i+1][0]
                merged.append([onset, 0, etype])
                if verbose:
                    log.debug('merged {} + {} = {}'.format(event, events[i+1], merged[-1]))
            else:
                # either we are not interested in audio onsets or there is none
                merged.append(event)
                if verbose:
                    log.debug('kept {}'.format(merged[-1]))
        # audio onsets (etype == 1000) are not copied
        if etype > 1111: # other events (keystrokes)
            merged.append(event)
            if verbose:
                log.debug('kept other {}'.format(merged[-1]))

    merged = np.asarray(merged, dtype=int)

    if inplace:
        stim_id = raw.ch_names.index(stim_channel)
        raw._data[stim_id,:].fill(0)     # delete data in stim channel
        raw.add_events(merged)

    return merged


def decode_beat_event_type(etype):
    # etype = 100000 + stimulus_id * 1000 + condition * 100 + cue + beat_count

    etype = int(etype)
    etype -= 100000

    stimulus_id = etype / 1000
    condition = (etype % 1000) / 100  # hundreds
    cue = (etype % 100) / 10          # tens
    beat_count = etype % 10           # last digit

    return stimulus_id, condition, cue, beat_count


def filter_beat_events(events, stimulus_ids='any', conditions='any', beat_counts='any', cue_value='any'):
#     print 'selected stimulus ids:', stimulus_ids
#     print 'selected conditions  :', conditions
#     print 'selected beat counts :', beat_counts
    filtered = list()

    for event in events:
        etype = event[2]
        stimulus_id, condition, cue, beat_count = decode_beat_event_type(etype)

        if (stimulus_ids == 'any' or stimulus_id in stimulus_ids) and \
                (conditions == 'any' or condition in conditions) and \
                (beat_counts == 'any' or beat_count in beat_counts) and \
                (cue_value == 'any' or cue == cue_value):
            filtered.append(event)

    return np.asarray(filtered)

def decode_trial_event_type(etype):
    stimulus_id = etype / 10
    condition = etype % 10
    return stimulus_id, condition

def filter_trial_events(events, stimulus_ids='any', conditions='any'):
#     print 'selected stimulus ids:', stimulus_ids
#     print 'selected conditions  :', conditions

    filtered = list()

    for event in events:
        etype = event[2]
        if etype >= 1000:
            continue

        stimulus_id, condition = decode_trial_event_type(etype)

        if (stimulus_ids == 'any' or stimulus_id in stimulus_ids) and \
                (conditions == 'any' or condition in conditions):
            filtered.append(event)

    return np.asarray(filtered)


def add_trial_cue_offsets(trial_events, meta, raw_info, debug=False):
    sfreq = raw_info['sfreq']

    n_processed = 0
    for stim_id in STIMULUS_IDS:
        offset = int(np.floor(meta[stim_id]['length_of_cue'] * sfreq))

        for cond in [1,2]: # cued conditions
            event_id = get_event_id(stim_id, cond)
            ids = np.where(trial_events[:,2] == event_id)
            log.debug('processing {} events with id {}, offset={}'.format(len(ids[0]),event_id, offset))

            for i in ids:
                if debug:
                    log.debug('before: {}'.format(trial_events[i]))
                trial_events[i,0] += offset
                if debug:
                    log.debug('after:  {}'.format(trial_events[i]))
            n_processed += len(ids[0])

    log.info('processed {} trials.'.format(n_processed))
    return trial_events


def remove_overlapping_events(events, tmin, tmax, sfreq):
    filtered = []
    sample_len = (tmax-tmin) * sfreq
    last_end = sample_len
    for event in events:
        if event[0] > last_end:
            filtered.append(event)
            last_end = event[0] + tmin + sample_len
    filtered = np.asarray(filtered)
    print('kept {} of {} events'.format(len(filtered), len(events)))
    return filtered