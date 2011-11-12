import os
import pipes
import shutil
import subprocess
import paver.easy
import paver.path

paver.easy.options(
    build_root = paver.path.path(os.path.abspath(os.path.dirname(__file__))) / "build",
    source_root = paver.path.path(os.path.abspath(os.path.dirname(__file__))),
    )

def modified_since(target, inputs):
    """Is the target less recent than any of the inputs?"""

    if target.exists():
        return target.mtime < max(f.mtime for f in inputs)
    else:
        return True

def bibtex(options, input_path):
    """Generate a BibTex .bbl file."""

    target_path = paver.path.path(paver.path.path(input_path.namebase + ".bbl").basename() + ".aux")

    environment = dict(os.environ.items())

    environment["BIBINPUTS"] = \
        ":".join([
            options.source_root,
            input_path.parent,
            "",
            ])
    environment["BSTINPUTS"] = \
        ":".join([
            options.source_root,
            "",
            ])

    if modified_since(target_path, [input_path]):
        subprocess.check_call([
                "bibtex",
                input_path.basename(),
                ],
            cwd = options.build_root,
            env = environment,
            )

    return target_path

def sweave(options, input_path):
    """Generate the paper sections via Sweave."""

    target_path = options.build_root / paver.path.path(input_path.namebase + ".tex").basename()

    if not modified_since(target_path, [input_path]):
        paver.easy.sh(
            "R CMD Sweave {0}".format(input_path),
            cwd = options.build_root,
            )

def pgf_sweave(options, input_path):
    """Generate the paper sections via pgfSweave."""

    target_path = options.build_root / paver.path.path(input_path.namebase + ".tex").basename()

    if modified_since(target_path, [input_path]):
        copied_path = options.build_root / input_path.basename()

        shutil.copyfile(input_path, copied_path)

        paver.easy.sh(
            "Rscript ../pgfsweave-script.R --pgfsweave-only {0}".format(
                pipes.quote(copied_path.basename()),
                ),
            cwd = options.build_root,
            )

def tikz_make_external(options, makefile_path):
    """(Re)build external TikZ graphics."""

    # hackishly fix the makefile; no idea why this step is necessary
    with open(makefile_path) as makefile_file:
        makefile = makefile_file.read()

    with open(makefile_path, "w") as makefile_file:
        makefile_file.write(makefile.replace("^^I", "\t"))

    # then rebuild the graphics
    subprocess.check_call(
        [
            "make",
            "-f",
            makefile_path,
            ],
        cwd = options.build_root,
        env = set_texinputs(options, dict(os.environ.items())),
        )

def set_texinputs(options, environment):
    environment["TEXINPUTS"] = \
        ":".join([
            str(options.source_root),
            str(options.source_root / "sections"),
            str(options.source_root / "figures"),
            str(options.build_root),
            "",
            ])

    return environment

@paver.easy.task
def prepare(options):
    """Prepare the build directory."""

    if not options.build_root.exists():
        options.build_root.mkdir()

@paver.easy.task
@paver.easy.needs("prepare")
def paper(options):
    """Build the paper."""

    # run the relevant inputs through Sweave
    sweave_names = [
        "discovery",
        "scaling",
        ]
    sweave_paths = [options.source_root / "sections" / (name + ".Rnw") for name in sweave_names]

    for path in sweave_paths:
        pgf_sweave(options, path)

    # then put everything together with XeLaTeX
    def xelatex():
        input_paths = [
            options.source_root / "writeup.tex",
            options.source_root / "references.bib",
            options.source_root / "sections/abstract.tex",
            options.source_root / "sections/introduction.tex",
            options.source_root / "sections/conclusion.tex",
            options.build_root / "writeup.bbl",
            ]

        input_paths += [options.build_root / (name + ".tex") for name in sweave_names]

        paper_target_path = options.build_root / "writeup.pdf"

        if modified_since(paper_target_path, input_paths):
            environment = set_texinputs(options, dict(os.environ.items()))

            subprocess.check_call([
                    "xelatex",
                    "-halt-on-error",
                    options.source_root / "writeup.tex",
                    ],
                cwd = options.build_root,
                env = environment,
                )

    xelatex()
    bibtex(options, options.build_root / "writeup.aux")
    bibtex(options, options.build_root / "writeup.aux")
    #tikz_make_external(options, options.build_root / "writeup.makefile")
    xelatex()
    xelatex()

