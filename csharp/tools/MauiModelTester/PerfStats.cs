namespace MauiModelTester
{
    internal class PerfStats
    {
        internal PerfStats()
        {
            _runTimes = new List<double>();
        }

        internal TimeSpan LoadTime { get; set; }
        internal TimeSpan WarmupTime { get; set; }

        /// <summary>
        /// Add TimeSpan for one call to Run.
        /// </summary>
        /// <param name="runTime">Elapsed time</param>
        internal void AddRunTime(TimeSpan runTime)
        {
            _runTimes.Add(runTime.TotalMilliseconds);
        }

        internal void ClearRunTimes()
        {
            _runTimes.Clear();
        }

        internal List<string> GetRunStatsReport(bool outputRunTimes = false)
        {
            List<string> lines = new List<string>();

            if (_runTimes.Count > 0)
            {
                _runTimes.Sort();

                var totalRunTime = _runTimes.Sum();

                lines.Add($"Total time for {_runTimes.Count} iterations: {totalRunTime:F4} ms\n");
                lines.Add($"Average run time: {totalRunTime / _runTimes.Count:F4} ms");
                lines.Add($"Minimum run time: {_runTimes.Min():F4} ms");
                lines.Add($"Maximum run time: {_runTimes.Max():F4} ms\n");
                lines.Add($"50th Percentile run time: {_runTimes[(int)(_runTimes.Count * 0.5)]:F4} ms");
                lines.Add($"90th Percentile run time: {_runTimes[(int)(_runTimes.Count * 0.9)]:F4} ms");

                if (_runTimes.Count >= 100)
                {
                    lines.Add($"95th Percentile run time: {_runTimes[(int)(_runTimes.Count * 0.95)]:F4} ms");
                    lines.Add($"99th Percentile run time: {_runTimes[(int)(_runTimes.Count * 0.99)]:F4} ms");
                }

                if (outputRunTimes)
                {
                    lines.Add("\nRun times (ms):");
                    lines.Add(string.Join(",", _runTimes.Select(x => x.ToString("F2"))));
                }
            }

            return lines;
        }

        private List<double> _runTimes;
    }
}
