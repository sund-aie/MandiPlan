"""The guided workflow: what is done, what is next, what is blocked."""

from __future__ import annotations

import numpy as np
import pytest

from make_phantom import PhantomSpec, make_phantom, write_dicom_series
from mandiplan.workflow import current_step, workflow_status


@pytest.fixture(scope="module")
def loaded(qt_app, tmp_path_factory):
    from mandiplan.ui.session import Session

    spec = PhantomSpec()
    folder = tmp_path_factory.mktemp("workflow_dicom")
    write_dicom_series(make_phantom(spec), folder)
    session = Session()
    session.load_dicom_folder(str(folder))
    return spec, session


def test_an_empty_session_asks_for_a_volume(qt_app):
    from mandiplan.ui.session import Session

    statuses = workflow_status(Session())
    assert len(statuses) == 5
    assert not any(status.done for status in statuses)
    assert statuses[0].available
    assert "Open a DICOM folder" in statuses[0].summary
    # Everything after step 1 is blocked, and says what it is waiting for.
    for status in statuses[1:]:
        assert status.blocked_by
        assert not status.available


def test_loading_a_volume_completes_the_first_step(loaded):
    _, session = loaded
    statuses = workflow_status(session)
    assert statuses[0].done
    assert "mm" in statuses[0].summary
    assert current_step(session).index == 1


def test_the_workflow_advances_as_the_plan_is_built(loaded):
    spec, session = loaded

    for point in spec.centre_line(9, extend_deg=12.0):
        session.add_arch_seed([point[0], point[1], spec.centre_z])
    assert workflow_status(session)[1].done
    assert current_step(session).index == 2

    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid - 12.0, +1.0), (mid + 12.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(
            session.frames.points[index], sign * session.frames.tangents[index]
        )
    resection = workflow_status(session)[2]
    assert resection.done
    assert "not executed yet" in resection.summary

    # Step 4 is now reachable, and names the button to press.
    reconstruction = workflow_status(session)[3]
    assert reconstruction.available
    assert "mid-sagittal" in reconstruction.summary

    session.estimate_symmetry_plane()
    session.build_graft()
    assert workflow_status(session)[3].done


def test_a_plate_that_does_not_fit_keeps_its_step_unfinished(loaded):
    spec, session = loaded
    radius = spec.radius_mm + spec.semi_axis_bl_mm + 0.5
    centre = np.radians(spec.arch_centre_deg)
    for theta in np.linspace(centre - np.radians(20), centre + np.radians(20), 8):
        session.add_plate_point(
            [radius * np.cos(theta), radius * np.sin(theta), spec.centre_z]
        )
    plate = workflow_status(session)[4]
    assert not plate.done
    assert "screw hole" in plate.summary

    session.clear_plate_path()
    for theta in np.linspace(centre - np.radians(62), centre + np.radians(62), 16):
        session.add_plate_point(
            [radius * np.cos(theta), radius * np.sin(theta), spec.centre_z]
        )
    assert workflow_status(session)[4].done
