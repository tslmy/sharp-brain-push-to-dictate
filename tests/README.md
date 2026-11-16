# Regression tests

No audio is bundled (the licence of text-to-speech output varies). Make a few
16 kHz mono 16-bit WAVs of your own, for example:

```sh
espeak-ng -w clip1.wav "Turn left at the next intersection."   # then: sox clip1.wav -r 16000 -c 1 -b 16 clips/t1.wav
mkdir -p clips && python3 mkref.py UsefulSensors/moonshine-tiny clips   # float32 reference transcripts
make -C .. test                                                  # compare the int8 engine
```

The int8 engine matches the reference exactly on most clips; a near-tie word
can differ, so the test only fails when more than two clips differ.
