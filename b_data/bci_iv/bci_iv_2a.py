import mne

raw = mne.io.read_raw_gdf('A01T.gdf')

for i in range(1, 9):
    fn = 'A' + '%02d' % (i + 1) + 'T.gdf'
    tem = mne.io.read_raw_gdf(fn)
    if raw:
        raw.append(tem)
        print(raw)
    else:
        print(fn)
        exit()

raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

event_dict = {'769': 0, '770': 1, '771': 2, '772': 3}
events, _ = mne.events_from_annotations(raw, event_id=event_dict)
epochs = mne.Epochs(raw, events, tmin=1 / 250, tmax=4, baseline=None, preload=True)
epochs.save('bci_iv_2a_epo.fif', overwrite=True)
