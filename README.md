 # CWT-based co-movement and active transport detection
 
This repository contains a workflow for detecting co-movement and active transport from 2D live-cell imaging data.

If you find the code posted here useful for any work you are going to publish, please cite [our paper where the workflow 
is published](https://www.mdpi.com/2073-4409/11/2/270/htm): Polev, K.; Kolygina, D.V.; Kandere-Grzybowska, K.; Grzybowski, B.A. Large-Scale, Wavelet-Based Analysis of Lysosomal Trajectories and Co-Movements of Lysosomes with Nanoparticle Cargos. *Cells* __2022__, *11*, 270. https://doi.org/10.3390/cells11020270

To install, you should have Python 3.6 or later (tested on Python 3.8). 
Run the following command to install the required packages:
`python3 -m pip install -r requirements.txt`

The code contains:
- `trackedcell.py`: initial data processing, wavelet-based co-movement and active transport detection.
- `dataprep.py`: further data processing, heavy-tailed models fitting and comparison using Akaike weights, MSD calculation and fitting, etc.
- `layout_utils.py`: plotting routines (including box plots with automated p-value calculation and display of test results in the plot).

More ready-to-use GUI pages and examples of usage are coming soon (the work on refactoring and cleaning the code is continued as the toolset is being expanded).

Some data example file is in folder `data example`.

### Usage example – an interactive page
Example in `frontpage.py` shows the typical workflow, with MSD fitting and detection of co-movement,
locally serving the app with GUI on your computer.
Simply run it with python (`python3 frontpage.py`) and open url http://127.0.0.1:13888/ in your web browser.
On the GUI page, you can upload .csv file with your data following this format:

| id  | posx | posy | ... |   |
|-----|------|------|-----|---|
| 1   | x1   | y1   |     |   |
| 1   | x2   | y2   |     |   |
| 1   | x3   | y3   |     |   |
| 2   | x1   | y1   |     |   |
| 2   | x2   | y2   |     |   |
| 2   | x3   | y3   |     |   |
| ... | ...  | ...  |     |   |
| 1   | x1   | y1   |     |   |
| ... |      |      |     |   |

The columns (with names) shown here are mandatory; `time` column is recommended, but should be auto-generated if not present
(assuming 5s intervals between the frames); additional columns might be present, but are not used.
`posx` stands for x-coordinate (consecutive in time), `posy` for y-coordinate, `id` for track id (consecutive, starting from 1).
To detect co-movement among two types of objects, ids of objects of type 1 are followed by ids of objects of type 2, 
as shown in the table above. Parameters for data processing are pre-set and can be changed in the code.

Keep in mind that the plots are interactive. Hovering your mouse above the plot will show the data point values. 
Clicking on the point from the co-moving pairs plot will show the corresponding objects' trajectories, animated in time.

The example code is a good starting point for your own projects. Most data processing is in the `process_data` function, 
and the pipeline can be expanded using routines from `dataprep.py`. The biggest part of the code is dash-based GUI; more 
GUI plotting functions are in `layout_utils.py`.
