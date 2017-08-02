# pytrack

object-oriented python particle pusher for ALPHA

Run me using (for example) "python run.py pbars 100" to generate one hundred tracks from the simulation scenario "pbars" 

# Changelog

First commit (02-08-2017)

An untested track generator for fully axisymmetric geometries.

Some notes about the current version:
- This version doesn't yet have a postprocessor component, and so the simulation accuracy hasn't yet been thoroughly tested.
User beware!

- The magnetic field map is hard-coded. This version will only accept axisymmetric field maps oriented along z.
