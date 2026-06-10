# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache_ascend.v1.transfer_channel.transfer_spec import TimedTransferEvent


class _FakeTimingEvent:
    def elapsed_time(self, other):
        assert other is not None
        return 125.0


def test_timed_transfer_event_reports_seconds_and_bytes_once():
    reports = []
    completion = TimedTransferEvent(
        event=object(),
        start_event=_FakeTimingEvent(),
        end_event=object(),
        callback=lambda duration, num_bytes: reports.append((duration, num_bytes)),
        num_bytes=4096,
    )

    completion.report()
    completion.report()

    assert reports == [(0.125, 4096)]
