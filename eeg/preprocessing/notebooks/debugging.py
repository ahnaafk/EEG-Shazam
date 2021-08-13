"""
General workflow for importing a new session
"""

subject = 'P01' # TODO: change this for each subject
verbose = True  # change this for debugging

import matplotlib

from pipeline import Pipeline
settings = dict(debug=False, mne_log_level='Info', sfreq=64) # optional pipeline settings
pipeline = Pipeline(subject, settings)

pipeline.load_raw(verbose=verbose)

print('done')


pipeline.mark_bad_channels(['P8', 'P10', 'T8'], save_to_raw=True)
pipeline.interpolate_bad_channels()


# pipeline.check_trial_events()
# pipeline.check_trial_audio_onset_merge(use_audio_onsets=True, verbose=None)
# pipeline.merge_trial_and_audio_onsets()

# pipeline.bandpass_filter()
# pipeline.generate_beat_events() # Note: this includes cue-beats !!!
# #pipeline.beat_epochs.average().plot();
# pipeline.find_eog_events()
# pipeline.alt_downsample()
# pipeline.check_resampled_trial_events()
# pipeline.compute_ica()