"""
``showerpipe.lhe``
==================

The ShowerPipe Les Houches functions utilise xml parsing techniques to
redistribute and repeat hard events, outputting valid lhe files.
"""

import contextlib as ctx
import functools as fn
import gzip as gz
import io
import itertools as it
import re
import shutil
import tempfile as tf
import typing as ty
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from xml.sax.saxutils import unescape

import typing_extensions as tyx
from lxml import etree

__all__ = ["source_adapter", "load_lhe", "count_events", "split", "LheData"]

_LHE_STORAGE: tyx.TypeAlias = ty.Union[Path, str, bytes]

_PARSE_KWARGS = {
    "resolve_entities": False,
    "remove_comments": False,
    "strip_cdata": False,
}


@ctx.contextmanager
def source_adapter(source: _LHE_STORAGE) -> ty.Iterator[io.BufferedIOBase]:
    """Context manager to provide a consistent adapter interface for LHE
    data stored in various formats.

    :group: LesHouches

    Parameters
    ----------
    source : Pathlike, string, or bytes
        The variable or filepath containing the LHE data. May be a path,
        url, string, or bytes object. Gzip compression is allowed.

    Returns
    -------
    lhe_file : io.BufferedIOBase
        File-like object containing the Les Houches data. If ``source``
        is a URL pointing to a non-gzipped file, ``lhe_file`` will not
        be seekable.
    """
    is_bytes = isinstance(source, bytes)
    is_str = isinstance(source, str)
    is_path = (is_str or isinstance(source, Path)) and Path(
        source
    ).exists()  # type: ignore
    is_url = False
    out_io: io.BufferedIOBase
    if is_str:
        is_url = bool(urlparse(source).netloc)  # type: ignore
    if is_path:
        path = Path(source)  # type: ignore
        try:
            with open(path) as lhe_filecheck:
                lhe_filecheck.read(1)
            out_io = open(path, "rb")
            yield out_io
        except UnicodeDecodeError:
            out_io = gz.open(path, "rb")
            out_io.read(1)
            out_io.seek(0)
            yield out_io
        finally:
            out_io.close()  # type: ignore
    elif is_str or is_bytes or is_url:  # create a BytesIO file-object
        if is_url:
            lhe_response = urlopen(source)  # type: ignore
            out_io = gz.GzipFile(fileobj=lhe_response, mode="rb")
            try:
                out_io.read(1)
                out_io.seek(0)
            except gz.BadGzipFile:
                lhe_response.close()
                out_io.close()
                out_io = urlopen(source)  # type: ignore
        else:
            if is_str:
                xml_bytes = source.encode()  # type: ignore
            else:
                xml_bytes: bytes = source  # type: ignore
            out_io = io.BytesIO(xml_bytes)
        try:
            yield out_io
        finally:
            out_io.close()
    else:
        raise NotImplementedError


def load_lhe(path: ty.Union[Path, str]) -> io.BytesIO:
    """Load Les Houches file into a bytestring object.

    :group: LesHouches

    Parameters
    ----------
    path : Path or str
        Location of the Les Houches file, either on disk, or as online
        as a URL.

    Returns
    -------
    BytesIO
        Buffer containing bytes content of the Les Houches file.
    """
    parser = etree.XMLParser(**_PARSE_KWARGS)
    with source_adapter(path) as file_in:
        tree = etree.parse(file_in, parser=parser)
    root = tree.getroot()
    return _root_to_bytes(root)


def _get_mg_info(header_element: etree.ElementBase) -> etree.ElementBase:
    return header_element.find("MGGenerationInfo")


def _read_num_events(mg_info: etree.ElementBase) -> int:
    re_num = re.findall(r"\d+", mg_info.text)
    num_events = int(re_num[0])
    return num_events


def _update_num_events(
    mg_info: etree.ElementBase, new_num: int
) -> etree.ElementBase:
    mg_info = deepcopy(mg_info)
    prev_num = _read_num_events(mg_info)
    mg_info_list = mg_info.text.split("\n")
    mg_info_list[1] = mg_info_list[1].replace(str(prev_num), str(new_num))
    mg_info.text = "\n".join(mg_info_list)
    return mg_info


def count_events(source: _LHE_STORAGE) -> int:
    """Returns the number of LHE events stored in ``source``.

    :group: LesHouches

    Parameters
    ----------
    source : pathlib.Path | str | bytes
        File or string object containing Les Houches data.

    Returns
    -------
    int
        The number of LHE events.
    """
    with source_adapter(source) as xml_source:
        event_parser = etree.iterparse(
            source=xml_source, tag=("event",), **_PARSE_KWARGS
        )
        num_events = 0
        for _, event in event_parser:
            num_events = num_events + 1
            event.clear()
    return num_events


def split(source: _LHE_STORAGE, stride: int) -> ty.Iterator[io.BytesIO]:
    """Generator, splitting LHE file content into separate buffers
    representing LHE files, with maximum number of events per split
    equal to stride.

    :group: LesHouches

    Parameters
    ----------
    source : Pathlike, string, or bytes
        The variable or filepath containing the LHE data. May be a path,
        string, or bytes object. The path may be compressed with gzip.

    Yields
    ------
    BytesIO
        Binary buffers, containing the content for the split Les Houches
        file.

    Notes
    -----
    Particularly useful for large LHE files, which cannot fit in memory.
    """
    with ctx.ExitStack() as stack:
        xml_source = stack.enter_context(source_adapter(source))
        if not xml_source.seekable():
            temp = stack.enter_context(tf.TemporaryFile())
            shutil.copyfileobj(xml_source, temp)
            temp.seek(0)
            xml_source = temp  # type: ignore
        lhe_root_tagname = "LesHouchesEvents"
        lhe_root_parser = etree.iterparse(
            source=xml_source,
            events=("start",),
            tag=(lhe_root_tagname,),
            **_PARSE_KWARGS,
        )
        _, lhe_root_meta = next(lhe_root_parser)
        lhe_root_template = etree.Element(lhe_root_tagname)
        lhe_root_template.set("version", lhe_root_meta.get("version"))

        xml_source.seek(0)
        header_parser = etree.iterparse(
            source=xml_source, tag=("header", "init"), **_PARSE_KWARGS
        )
        _, header = next(header_parser)
        _, init = next(header_parser)
        lhe_root_template.append(header)
        lhe_root_template.append(init)

        mg_info = _get_mg_info(header)
        total_events = _read_num_events(mg_info)

        splits = [stride for _ in range(total_events // stride)]
        split_leftover = total_events % stride
        if split_leftover > 0:
            splits = splits + [split_leftover]

        xml_source.seek(0)
        event_parser = etree.iterparse(
            source=xml_source, tag=("event",), **_PARSE_KWARGS
        )

        for split_ in splits:
            lhe_root = deepcopy(lhe_root_template)
            split_header = lhe_root[0]
            split_mg_info = _get_mg_info(split_header)
            split_header.replace(
                split_mg_info,
                _update_num_events(split_mg_info, new_num=split_),
            )
            for _ in range(split_):
                _, event = next(event_parser)
                lhe_root.append(deepcopy(event))
                event.clear(keep_tail=True)
            content = io.BytesIO(etree.tostring(lhe_root))
            content.seek(0)
            yield content


def _root_to_bytes(root: etree.ElementBase) -> io.BytesIO:
    content_invalid = etree.tostring(root)

    def unescape_bytes(x: bytes) -> bytes:
        return unescape(x.decode()).encode()

    content = io.BytesIO(unescape_bytes(content_invalid))
    content.seek(0)
    return content


class LheData:
    """Container for the Les Houches file content.

    :group: LesHouches

    Parameters
    ----------
    content : bytes or BinaryIO or str or Path
        Content of the Les Houches file. May be passed as a bytestring
        or file-like BinaryIO buffer. Can also be read from disk by
        passing a path, as either a string or pathlib.Path instance.
    """

    def __init__(
        self, content: ty.Union[bytes, str, Path, io.BufferedIOBase]
    ) -> None:
        if isinstance(content, bytes):
            content_bytes = content
        elif isinstance(content, io.BufferedIOBase):
            content.seek(0)
            content_bytes = content.read()
        else:
            with open(content, mode="rb") as content_file:
                content_bytes = content_file.read()
        self._root: etree.ElementBase = etree.fromstring(content_bytes)

    def __repr__(self) -> str:
        return f"LheData(num_events={self.num_events})"

    @classmethod
    def from_storage(cls, storage: ty.Union[str, Path]) -> "LheData":
        """Loads the LHE data directly from the given file location.

        Parameters
        ----------
        storage : str, Path
            File location. Can be string, path, or a URL.

        Returns
        -------
        lhe_data : LheData
            Instance of LheData loaded with the data from the given
            ``storage``.
        """
        return cls(load_lhe(storage).getvalue())

    @property
    def content(self) -> io.BytesIO:
        """Buffer containing the content of the Les Houches file."""
        return _root_to_bytes(self._root)

    @content.setter
    def content(self, data: bytes) -> None:
        """Overwrite the content buffer with a bytestring."""
        del self.num_events
        self._root = etree.fromstring(data)

    @fn.cached_property
    def num_events(self) -> int:
        """Counts the number of events within a LHE file."""
        return len(self._root.findall("event"))

    @property
    def _event_iter(self) -> ty.Iterator[etree.ElementBase]:
        return self._root.iter("event")  # type: ignore

    def repeat(
        self: tyx.Self, repeats: int, inplace: bool = False
    ) -> tyx.Self:
        """Modifies LHE content, repeating each event the number of
        times given by repeats.

        Parameters
        ----------
        repeats : int
            The number of times to repeat the events.
        inplace : bool
            If True, modifies content inplace and returns None.

        Returns
        -------
        LheData
            LheData instance populated with repeated events.

        Notes
        -----
        Repeat means, eg. repeat([A, B, C], 2) => [A, A, B, B, C, C]

        Modifying inplace is more computationally efficient than setting
        the content explicitly to bytestring output, but may lead to
        unexpected side effects. Should be used with caution.
        """
        return self._event_duplicator(
            repeats, inplace, dup_strat=self._repeat_order
        )

    def tile(self: tyx.Self, repeats: int, inplace: bool = False) -> tyx.Self:
        """Modifies LHE content, tile repeating all events the
        number of times given by repeats.

        Parameters
        ----------
        repeats : int
            The number of times to repeat the events.
        inplace : bool
            If True, modifies content inplace and returns None.

        Returns
        -------
        LheData
            LheData instance populated with repeated events.

        Notes
        -----
        Tile repeat means, eg. tile([A, B, C], 2) => [A, B, C, A, B, C].

        Modifying inplace is more computationally efficient than setting
        the content explicitly to bytestring output, but may lead to
        unexpected side effects. Should be used with caution.
        """
        return self._event_duplicator(
            repeats, inplace, dup_strat=self._tile_order
        )

    def _tile_order(
        self, x: ty.Iterator[etree.ElementBase]
    ) -> ty.Iterator[etree.ElementBase]:
        return x

    def _repeat_order(
        self, x: ty.Iterator[etree.ElementBase]
    ) -> ty.Iterator[etree.ElementBase]:
        return zip(*x)

    def _build_root(
        self, event_iter: ty.Iterator[etree.ElementBase]
    ) -> etree.ElementBase:
        root = deepcopy(self._root)
        for event in root.findall("event"):
            root.remove(event)
        for event in event_iter:
            root.append(event)
        return root

    def _event_duplicator(
        self: tyx.Self,
        repeats: int,
        inplace: bool,
        dup_strat: ty.Callable[
            [ty.Iterator[etree.ElementBase]], ty.Iterator[etree.ElementBase]
        ],
    ) -> tyx.Self:
        tiled_lists = (self._event_iter for _ in range(repeats))
        dup_events = it.chain.from_iterable(dup_strat(tiled_lists))
        dup_event_copies = map(deepcopy, dup_events)
        root = self._build_root(dup_event_copies)
        header = root[0]
        mg_info = _get_mg_info(header_element=header)
        prev_num = self.num_events
        new_num = prev_num * repeats
        header.replace(mg_info, _update_num_events(mg_info, new_num))
        if inplace is True:
            self._root = root
            if self.num_events is not None:
                del self.num_events
            return self
        return type(self)(_root_to_bytes(root).getvalue())
