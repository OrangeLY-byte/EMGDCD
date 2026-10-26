# Bundled NonLinearVAR generator

Source: vae-multi-level-neural-GC-official-master (the local upstream checkout).
License: GNU Affero General Public License v3; see LICENSE.

sim_oneband.py and configs/NonLinearVAR.yaml are unmodified upstream copies.
The simulator package initializer exports only simOneBanded2VAR, which is used
by MSCD_nonlinvar_independent.py. Unrelated simulators are not bundled.

The experiment defaults to this directory and requires no external checkout.
Its existing zero-initialized trajectory simulation remains in the experiment
script. Use --generator-root PATH to explicitly select another generator root.