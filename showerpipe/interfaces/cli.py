import click

from showerpipe.generator import PythiaGenerator
from showerpipe.interfaces._subject import ShowerPipeline
from showerpipe.interfaces import observer


@click.command()
@click.argument('settings-file', type=click.Path(exists=True))
@click.option('--me-file', type=click.Path(exists=True))
@click.option('--hdf/--no-hdf', default=False)
def main(settings_file, me_file, hdf):
    data_generator = PythiaGenerator(
            config_file=settings_file,
            me_file=me_file,
            )
    pipeline = ShowerPipeline(data_generator)
    if hdf:
        hdf_observer = observer.HdfStorage()
        pipeline.attach(hdf_observer)
    with click.progressbar(pipeline) as events:
        for event in events:
            event.notify()
