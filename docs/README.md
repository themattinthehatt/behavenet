Welcome! This guide walks you through how to automatically generate 
documentation for this python project using the 
[sphinx](http://www.sphinx-doc.org/en/stable/index.html) package in python, and 
how to publish it on 
[Read the Docs](https://readthedocs.org/)
so that users can easily access and search your documentation.

## 1. INSTALL SPHINX (general instructions [here](http://www.sphinx-doc.org/en/stable/tutorial.html))

### Install Sphinx from the command line:

```
$ pip install Sphinx
```

### sphinx-quickstart

Next we want to set up the source directory for the documetnation. In the 
command line, `cd` to the root of the project directory and enter:

```
$ sphinx-quickstart
```

You'll be prompted to enter a number of user options. For most you can just 
accept the defaults, but you'll want to change the following:

* root path for documentation: `./docs`
* `autodoc`: y (allows automatic parsing of docstrings)
* `viewcode`: y (links documentation to source code)
* `mathjax`: y (allows mathjax in documentation)
* `githubpages`: y (allows integration with github)

## 2. SET UP CONF.PY FILE
Now that sphinx is installed, we want to configure it to automatically parse our
meticulously maintained docstrings and generate html pages that display said 
information in a readable and searchable way. 
More details on the Google-style python docstrings used in this project can be 
found 
[here](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

### Set project path
In the `conf.py` file, uncomment the following lines at the top so that the conf 
file (located in `./docs`) can find the project (located above in `./`):

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
```

### Theme
Change the theme to something nicer than the default:

```python
html_theme = 'sphinx_rtd_theme'
```

Find the themes available through sphinx 
[here](http://www.sphinx-doc.org/en/stable/theming.html).

### Allow parsing of google-style docstrings
Add `sphinx.ext.napoleon` to the `extensions` variable in `conf.py`, so that it
looks something like this:

```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon'] 
```

### Include documentation for class constructors
If you want to document `__init__()` functions for python classes, add the 
following functions to the end of the `conf.py` file 
(thanks to https://stackoverflow.com/a/5599712):

```python
def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
```

### Include inherited attributes and methods in documentation
This can help if you want users to be able to find all available attributes and
methods, including those that are inherited, for python classes. Add
`:inherited-members:` to each module in the `./docs/source/*.rst` files. For 
example, to show attributes and methods that the `NDN.network` module inherits 
from its base classes, the `network` module in the `./docs/source/NDN.rst` file 
should look like:

```
NDN\.network module
-------------------

.. automodule:: NDN.network
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
```

### Get autodocs working
In the command line, from the `./docs` directory, run:

```
$ sphinx-apidoc -o source/ ../
```

For some reason this is necessary to get autodocs working.

### Build the documentation
In the directory, run:

```
$ make html
```

You'll then be able to find the documentation landing page at
`/docs/_build/html/index.html`

## 3. ADD A NEW PAGE 
Docstrings are useful for understanding how individual functions work, but do 
not help too much for a new user of the code. To facilitate learning how the
code works we will want to create tutorial pages that demonstrate how to use 
certain features of the code.

In the directory containing the `index.rst` file, add a new file called 
`tutorial-example.rst`. This will look something like:

```
################
Tutorial Example
################

Here's some content written in reStructured Text (.rst), a markup language 
commonly used for technical documentation
```

Tell sphinx where this file is by adding `tutorial-example` to the 
`.. toctree::` section in the `index.rst` file, so that it looks something like
this:

```
.. toctree::
   :maxdepth: 2

   tutorial-example
   another-tutorial-example
```

## 4. PUBLISH THE DOCUMENTATION (general instructions [here](http://dont-be-afraid-to-commit.readthedocs.io/en/latest/documentation.html))
Now that we've built our documentation, we want to publish it on the web. 
Fortunately, Read the Docs and GitHub make this super simple. The following steps
are mostly copy-and-pasted from the general instructions above.

### Exclude unwanted directories
We do not want to commit the rendered files to github, just the source. To 
exclude these, add them to `.gitignore`:

```
_build
_static
_templates
```

Then push the updated files to GitHub.

### Create an account with readthedocs.org
Follow the instructions there, they should be self-explanatory.

And now, just like magic, Read the Docs will watch your GitHub project and 
update the documentation every night.

But wait! You can do better, if you really think it is necessary. On GitHub:
1. select **settings** for your project (not for your account!) in the 
navigation panel on the right-hand side
2. choose **Webhooks & Services**
3. enable `ReadTheDocs` under **Add Service** dropdown

...and now, every time you push documents to GitHub, ReadTheDocs will be 
informed that you have new documents to be published. It's not magic, they say,
but it's pretty close. Close enough for me.

