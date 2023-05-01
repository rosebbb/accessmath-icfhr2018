1. Installation
    ```
    conda install numpy
    pip install opencv-python
    conda install -c anaconda scipy
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install -c cogsci pygame
    ```

1. Data annotation

    a. Convert xml to coco
        * Run `utils/xml2coco.py`
        * Saved at `/data/Projects/accessmath-icfhr2018/AccessMathVOC` 

    b. Set up CVAT - done

    c. Error when uploading annotation json file to CVAT



2. Binarization

    a. Run code:
        `python pre_ST3D_v2.0_04_td_ref_binarize.py`

        