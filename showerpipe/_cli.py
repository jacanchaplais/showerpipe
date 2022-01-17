import click

from showerpipe.generator import PythiaGenerator
from showerpipe.interfaces._subject import ShowerPipeline
from showerpipe.interfaces import observer


def run(pipeline, listeners):
    """Attaches the observers to the event generator, and then runs it."""
    for listen in listeners: pipeline.attach(listen)
    with click.progressbar(pipeline) as events:
        for event in events: event.notify()

@click.group()
@click.option(
        '--hdf-path', type=click.Path(dir_okay=False, writable=True),
        default='',
        help='save as hdf at specified path (extension .h5 or .hdf5)'
        )
@click.pass_context
def showergen(ctx, hdf_path):
    """Generate Monte-Carlo hadronisation and particle shower data
    in an extensible pipeline.
    """
    ctx.obj = {}
    ctx.obj['listeners'] = [] # prevent ns collision with observer module
    if hdf_path != '':
        ctx.obj['listeners'].append(observer.HdfStorage(hdf_path))

@showergen.command()
@click.argument('settings-file', type=click.Path(exists=True))
@click.option(
        '--me-file', type=click.Path(exists=True),
        help='Les Houches file containing hard events.'
        )
@click.pass_context
def pythia(ctx, settings_file, me_file):
    """Runs Pythia using SETTINGS_FILE, typically with extension 'cmnd'."""
    data_generator = PythiaGenerator(
            config_file=settings_file,
            me_file=me_file,
            )
    pipeline = ShowerPipeline(data_generator)
    run(pipeline, ctx.obj['listeners'])
