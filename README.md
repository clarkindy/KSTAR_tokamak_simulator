# KSTAR tokamak simulator (KSTAR-NN)

> [!NOTE]
>
> This is an unofficial Streamlit port of KSTAR_tokamak_simulator. You might want to check out [the original repository](https://github.com/jaem-seo/KSTAR_tokamak_simulator).
>
> The key differences to the original version are as follows:
>
> - When auto-run is enabled, the slider should be stopped for a moment for the changed value to be sent to the simulator. This leads to slower running speed; the demo video is accelerated to 4x the original speed.
> - The set of variables and their orders are changed. This version shows $`H_{98(y,2)}`$ instead of $`\beta_p`$.
> - `Dump outputs` button is replaced by `Toggle display` button, which shows the table of previously dumped variables.
> - The checkboxes `Plot NBI/EC` and `Plot heat load` are removed. You can toggle their display by clicking on the respective parts of the legend.

- KSTAR is a tokamak (donut-shaped nuclear fusion device) located in South Korea.
- This repository provides a KSTAR tokamak simulation tool with LSTM-based neural network.
- See also [AI Tokamak Control](https://github.com/jaem-seo/AI_tokamak_control) where the AI replaces the manual control of this simulator.

# Installation
- You can install by
```console
$ git clone https://github.com/clarkindy/KSTAR_tokamak_simulator.git
$ cd KSTAR_tokamak_simulator
$ git switch testing/clarkindy
```

# Try it out
- [uv](https://docs.astral.sh/uv) is recommended to run this app. Open the GUI by typing below. It might take a bit depending on your environment.
```console
$ uv run streamlit run kstar_simulator.py
```
<p align="center">
  <img src="/images/gui.png">
</p>

- Slide the toggles in the left side and see the fusion plasma evolution in the right side.
<p align="center">
  <img src="/images/demo.gif">
</p>

- I hope you get insight with this virtual experiment!

# Note
- This simulation has been tested with many real discharges, and shows acceptable prediction accuracy.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165522817-bc56771f-600b-4c7c-a9c3-4da0256bfe3e.png">
</p>

- But it does not always guarantee perfect prediction since it doesn't account for all unknown factors.
- For example, the experiments #18672 and #22671 were conducted under almost the same setting, but showed quite different behaviors.
- In this case, the simulation shows quite a reasonable, average prediction as shown below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165521918-bd6969bf-31e0-4bf8-8848-f6ee6afeefaa.png">
</p>


# License
```
MIT License

Copyright (c) 2022 Jaemin Seo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
© 2022 GitHub, Inc.
Terms
```

# References
- J. Seo, et al. "Feedforward beta control in the KSTAR tokamak by deep reinforcement learning." Nuclear Fusion [61 (2021): 106010.](https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta)
- J. Seo, et al. "Development of an operation trajectory design algorithm for control of multiple 0D parameters using deep reinforcement learning in KSTAR." Nuclear Fusion [62 (2022): 086049.](https://iopscience.iop.org/article/10.1088/1741-4326/ac79be/meta)
