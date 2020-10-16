Training a model with multiple datasets
=======================================

The statistical models that comprise BehaveNet - autoencoders, ARHMMs, neural network decoders -
often require large amounts of data to avoid overfitting. While the amount of data collected in an
hour long experimental session may suffice, every one of these models will benefit from additional
data. If data is collected from multiple experimental sessions, and these data are similar enough
(e.g. same camera placement/contrast across sessions), then you can train BehaveNet models on all
of this data simultaneously.

BehaveNet provides two methods for specifying the experimental sessions used to train a model:

* :ref:`Method 1<all_keyword>`: use all sessions from a specified animal, experiment, or lab
* :ref:`Method 2<sessions_csv>`: specify the sessions in a csv file

The first method is simpler, while the second method offers greater control. Both of these methods
require modifying the data configuration json before training. We'll use the Musall dataset as an
example; below is the relevant section of the json file located in
``behavenet/configs/data_default.json`` that we will modify below.

.. code-block:: json

    "lab": "musall", # type: str

    "expt": "vistrained", # type: str

    "animal": "mSM30", # type: str

    "session": "10-Oct-2017", # type: str

    "sessions_csv": "", # type: str, help: specify multiple sessions

    "all_source": "save", # type: str, help: "save" or "data"

The Musall dataset provided with the repo (see ``behavenet/example/00_data.ipynb``) contains
autoencoders trained on two sessions individually, as well as a single autoencoder trained on both
sessions as an example of this.


.. _all_keyword:

Method 1: the "all" keyword
---------------------------
This method is appropriate if you want to fit a model on all sessions from a specified animal,
experiment, or lab. For example, if we want to fit a model on all sessions from animal
``mSM30``, we would modify the ``session`` parameter value to ``all``:

.. code-block:: json

    "lab": "musall", # type: str

    "expt": "vistrained", # type: str

    "animal": "mSM30", # type: str

    "session": "all", # type: str

    "sessions_csv": "", # type: str, help: specify multiple sessions

    "all_source": "save", # type: str, help: "save" or "data"

In this case the resulting models will be stored in the directory
``save_dir/musall/vistrained/mSM30/multisession-xx``, where ``xx`` can change. BehaveNet will
create a csv file named ``session_info.csv`` inside the multisession directory that lists the
lab, expt, animal, and session for all sessions in that multisession.


If we want to fit a model on all sessions from all animals in the ``vistrained`` experiment, we
would modify the ``animal`` parameter value to ``all``:

.. code-block:: json

    "lab": "musall", # type: str

    "expt": "vistrained", # type: str

    "animal": "all", # type: str

    "session": "all", # type: str

    "sessions_csv": "", # type: str, help: specify multiple sessions

    "all_source": "save", # type: str, help: "save" or "data"

In this case the resulting models will be stored in the directory
``save_dir/musall/vistrained/multisession-xx``. The string value for ``session`` does not
matter; BehaveNet searches for the ``all``
keyword starting at the lab level and moves down; once it finds the ``all`` keyword it ignores all
further entries.

.. note::

    The ``all_source`` parameter in the json file is included to resolve an ambiguity with the
    "all" keyword. For example, let's assume you use ``all`` at the session level for a single
    animal. If data for 6 sessions exist for that animal, and BehaveNet models have been fit to 4
    of those 6 sessions, then setting ``"all_source": "data"`` will use all 6 sessions with data.
    On the other hand, setting ``"all_source": "save"`` will use all 4 sessions that have been
    previously used to fit models.

.. _sessions_csv:

Method 2: session_info.csv file
--------------------------------
This method is appropriate if you want finer control over which sessions are included; for example,
if you want all sessions from one animal, as well as all but one session from another animal. To
specify these sessions, you can construct a csv file with the four column headers ``lab``,
``expt``, ``animal``, and ``session``. You can then provide this csv file (let's say it's called
``data_dir/example_sessions.csv``) as the value for the ``sessions_csv`` parameter:

.. code-block:: json

    "lab": "musall", # type: str

    "expt": "vistrained", # type: str

    "animal": "all", # type: str

    "session": "all", # type: str

    "sessions_csv": "data_dir/example_sessions.csv", # type: str, help: specify multiple sessions

    "all_source": "save", # type: str, help: "save" or "data"

The ``sessions_csv`` parameter takes precedence over any values supplied for ``lab``, ``expt``,
``animal``, ``session``, and ``all_source``.
