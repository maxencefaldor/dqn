#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import base64
from IPython.display import display, HTML


def ipython_show_video(path):
    """Show a video within IPython Notebook.
    
    Args:
        path: path of the video
    """
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>
        """.format(encoded.decode("ascii"))))
