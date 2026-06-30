
# Redirect stdout to stderr to enable output capture for pytest-xdist
import sys
import pytest
from py.xml import html
import re
import os
import platform
from datetime import datetime

def pytest_addoption(parser):
    parser.addoption("--configs", nargs='*', default=[], help='Path to config.yaml files')
    parser.addoption("--models", nargs='*', default=[], help='Filter models in the provided config files.')

    parser.addoption("--run-infer", action="store_true", default=False, help='Run Inference')
    parser.addoption("--disable-tidl-offload", action="store_true", help='Disable TIDL Offload')

    parser.addoption("--artifacts-dir", type=str, default=None, help='Directory to store/use compiled models artifacts. Default ./model-artifacts')
    parser.addoption("--reports-dir", type=str, default="reports", help='Directory to store resultant reports')

    parser.addoption("--force-runtime", type=str, default=None, help='Overwrite runtime defined in config file.')
    parser.addoption("--options", nargs='*', default=[[]], action="append", help='Space seperated extra compile or infer options. Ex: --options tensor_bits=8 advanced_options:quantization_scale_type=4')
    parser.addoption("--nmse-threshold", type=float, default=-1, help='NMSE threshold for inference testing')

    parser.addoption("--expected-fails", nargs='*', default=[], help='Space seperated expected failure tests.')
    parser.addoption("--disable-plot", action="store_true", default=False, help='Disable output plot in generated report')
    parser.addoption("--no-subprocess", action="store_true", default=False, help='Disable Running as subprocess')
    parser.addoption("--exit-on-critical-error", action="store_true", default=False, help='Force exit test on critical error')
    parser.addoption("--num-frames", type=int, default=None, help='Number of frames to run. Overwrites num_frames in model config if specified.')
    parser.addoption("--keep-full-model-artifacts", action="store_true", default=False, help='Do not remove tempDir from artifacts folder after compilation.')

def pytest_sessionfinish(session):
    try:
        plugin = session.config._json_report
        json_path = session.config.option.htmlpath.replace(".html",".json")
        plugin.save_report(json_path)
    except:
        pass

# Configures html report name and path
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    reports_dir = config.option.reports_dir
    os.makedirs(reports_dir, exist_ok=True)
    config.option.htmlpath = os.path.join(reports_dir, 'report_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")+".html")

    # Set EVM (aarch64) default timeout to 10s; x86 keeps pytest-timeout's default
    # Only applies when --timeout is not explicitly passed on the command line
    if platform.machine() == 'aarch64':
        cli_args = config.invocation_params.args
        if not any(arg.startswith('--timeout') for arg in cli_args):
            config.option.timeout = 10

    # Scale timeout by num_frames if provided, capped at 300s
    num_frames = config.option.num_frames
    if num_frames is not None:
        config.option.timeout = min(config.option.timeout * num_frames, 300)

# Adds the tidl_subgraphs attribute to test report
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    exit_on_critical_error = item.funcargs['exit_on_critical_error']
    #runtime = item.funcargs['runtime']
    runtime = "onnxrt"
    report.tidl_subgraphs = "Not detected"
    report.tidl_nodes = "Not detected"
    report.nmse = "-"
    report.mse = "-"
    report.max_delta = "-"
    report.post_proc_metrics = "-"
    report.plot_data = None
    report.perf_metrics = None

    runtime = "onnxrt"
    if hasattr(report, 'capstdout') and report.capstdout:
        runtime_regex = re.search(r'RUNTIME: (\w+)', report.capstdout)
        if runtime_regex:
            runtime = runtime_regex.group(1)
    report.runtime = runtime

    if report.when == 'call' or report.when == 'teardown':
        # Parsing subgraphs
        if runtime == "onnxrt" or runtime == "tflitert":
            if runtime == "onnxrt":
                num_subgraph_regex = re.search("Final number of subgraphs created are : ([0-9]*)", report.capstdout)
            else:
                num_subgraph_regex = re.search("Number of subgraphs:([0-9]*)", report.capstdout)

            if(num_subgraph_regex is None):
                c7x_table_regex = re.search(r"\|\s*C7x\s*\|\s*\d+\s*\|\s*(\d+|x)\s*\|", report.capstdout)
                if (c7x_table_regex is not None):
                    report.tidl_subgraphs = c7x_table_regex[1]
            else:
                report.tidl_subgraphs = num_subgraph_regex[1]

        elif runtime == "tvmrt":
            num_sg = 0
            # If import succeeded, extract from the success print
            tidl_import_regex = re.search("TIDL import of ([0-9]*) Relay IR subgraphs succeeded.",report.capstdout)
            if tidl_import_regex is not None:
                num_sg = tidl_import_regex[1]

            # This prints detected subgraphs from the IRModule before import starts. If import fails, extract detected sub graphs
            tvm_relay_detect = re.search("TVM Relay detected ([0-9]*) subgraphs", report.capstdout)
            if (tvm_relay_detect is not None and not num_sg):
                num_sg = tvm_relay_detect[1]

            # If all else fails, extract from performance summary (only printed during inference)
            num_subgraph_regex = re.search(r"num_subgraphs\s*:\s*([0-9]*)", report.capstdout)
            if (num_subgraph_regex is not None and not num_sg):
                num_sg = num_subgraph_regex[1]

            if num_sg:
                report.tidl_subgraphs = num_sg

        # Parsing nodes
        if (report.tidl_subgraphs.isdigit() and int(report.tidl_subgraphs) >= 1):
            if runtime == "tflitert":
                total_nodes_regex = re.search("out of ([0-9]*) nodes", report.capstdout)
                offloaded_nodes_regex = re.search("([0-9]*) nodes delegated", report.capstdout)
            else:
                total_nodes_regex = re.search("Total Nodes - ([0-9]*)", report.capstdout)
                offloaded_nodes_regex = re.search("Offloaded Nodes - ([0-9]*)", report.capstdout)

            if total_nodes_regex is not None and offloaded_nodes_regex is not None:
                try:
                    total_nodes = int(total_nodes_regex[1].strip())
                    offloaded_nodes = int(offloaded_nodes_regex[1].strip())
                    report.tidl_nodes = f"{offloaded_nodes}/{total_nodes}"
                except:
                    pass
            else:
                c7x_nodes_regex = re.search(r"\|\s*C7x\s*\|\s*(\d+)\s*\|", report.capstdout)
                cpu_nodes_regex = re.search(r"\|\s*CPU\s*\|\s*(\d+)\s*\|", report.capstdout)
                if c7x_nodes_regex is not None:
                    c7x_nodes = int(c7x_nodes_regex[1])
                    cpu_nodes = int(cpu_nodes_regex[1]) if cpu_nodes_regex is not None else 0
                    report.tidl_nodes = f"{c7x_nodes}/{c7x_nodes + cpu_nodes}"

        nmse_regex = re.search(r'MAX_NMSE: (\d*\.\d+|\d+|None)', report.capstdout)
        if nmse_regex:
            nmse = nmse_regex.group(1)
            report.nmse = str(nmse)
        mse_regex = re.search(r'MAX_MSE: (\d*\.\d+|\d+|None)', report.capstdout)
        if mse_regex:
            mse = mse_regex.group(1)
            report.mse = str(mse)
        max_delta_regex = re.search(r'MAX_DELTA: (\d*\.\d+|\d+|None)', report.capstdout)
        if max_delta_regex:
            max_delta = max_delta_regex.group(1)
            report.max_delta = str(max_delta)

        post_proc_metrics_regex = re.search(r'POST-PROC METRICS: (.+)$', report.capstdout, re.MULTILINE)
        if post_proc_metrics_regex:
            post_proc_metrics = post_proc_metrics_regex.group(1)
            report.post_proc_metrics = str(post_proc_metrics)

        # Parse performance metrics from the first TIDL inference block only.
        # Truncate at reference run marker to avoid picking up the CPU-only reference run.
        _ref_marker = "Generating reference outputs"
        _stdout_tidl = report.capstdout.split(_ref_marker)[0]
        _perf_block_match = re.search(
            r'={5,}\nAverage performance metrics[^\n]*\n-{5,}\n(.*?)\n={5,}',
            _stdout_tidl, re.DOTALL
        )
        if _perf_block_match:
            _block = _perf_block_match.group(1)
            def _parse_perf_field(name):
                m = re.search(rf'^\s*{re.escape(name)}\s*:\s*([\d.]+\s*\w+)', _block, re.MULTILINE)
                return m.group(1).strip() if m else None

            total_time = _parse_perf_field('total_time')
            core_time = _parse_perf_field('core_time')
            _sg = _parse_perf_field('subgraph_time')
            subgraph_time_label = 'subgraph_time' if _sg else 'graph_time'
            subgraph_time = _sg or _parse_perf_field('graph_time')
            read_total = _parse_perf_field('read_total')
            write_total = _parse_perf_field('write_total')
            ddr_total = _parse_perf_field('total')

            report.perf_metrics = {
                'total_time': total_time,
                'core_time': core_time,
                'subgraph_time': (subgraph_time_label, subgraph_time),
                'read_total': read_total,
                'write_total': write_total,
                'ddr_total': ddr_total,
            }
        else:
            _wall_time_match = re.search(r'WALL_TIME:\s*([\d.]+\s*\w+)', _stdout_tidl)
            if _wall_time_match:
                report.perf_metrics = {'wall_time': _wall_time_match.group(1).strip()}

        # Extract plot data from the output
        plot_data_regex = re.search(r'PLOT_BASE_64_PATH: (.+?)(?:\n|$)', report.capstdout)
        if plot_data_regex:
            plot_base64_path = plot_data_regex.group(1)
            try:
                with open(plot_base64_path, 'r') as f:
                    report.plot_data = f.read().strip()
            except Exception as e:
                pass

            if report.when == 'teardown':
                try:
                    dirname = os.path.dirname(plot_base64_path)
                    os.remove(plot_base64_path)
                    if not os.listdir(dirname):
                        os.rmdir(dirname)
                except Exception as e:
                    pass

        if exit_on_critical_error:
            ignore_filters = ["VX_ZONE_ERROR:Enabled","Globally Enabled","Globally Disabled","VX_ZONE_ERROR:[tivxObjectDeInit"]
            critical_errors = ["VX_ZONE_ERROR","dumped core","core dump","Segmentation fault","PROCESS TIMED OUT"]
            exit_msg = None
            for i in report.capstdout.strip().split('\n'):
                i = i.strip()
                ignore = False
                for j in ignore_filters:
                    if j in i:
                        ignore = True
                        break
                if not ignore:
                    for j in critical_errors:
                        if j in i:
                            exit_msg = f"CRITICAL_ERROR - {item.nodeid} - {j} detected. Exiting test run."
                            break
                if exit_msg:
                    break
            if exit_msg:
                report._exit_after_log = exit_msg

# Calls pytest.exit() after pytest-html has saved the report, so the output logs are captured
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_logreport(report):
    yield  # Let pytest-html and other plugins process the report first
    if hasattr(report, '_exit_after_log'):
        pytest.exit(report._exit_after_log)

# Inserts the TIDL Subgraphs table header
def pytest_html_results_table_header(cells):
    # Remove Links (index 3) and Duration (index 2)
    if len(cells) > 3:
        cells.pop(3)
    cells.pop(2)
    cells.insert(2, html.th("TIDL Offload Status"))
    cells.insert(3, html.th("Perf Metrics"))
    cells.insert(4, html.th("Output Metrics"))
    cells.insert(5, html.th("Output Plot"))

# Inserts the number of TIDL subgraphs for each row
def pytest_html_results_table_row(report, cells):
    if len(cells) > 3:
        cells.pop(3)
    cells.pop(2)

    # TIDL Offload Status: ALL / PARTIAL / NONE with node/subgraph counts
    if hasattr(report, 'tidl_subgraphs') and hasattr(report, 'tidl_nodes'):
        sgs = report.tidl_subgraphs
        nodes = report.tidl_nodes
        if nodes != "Not detected":
            try:
                offloaded, total = (int(x) for x in nodes.split('/'))
                if offloaded == 0:
                    status = "NONE"
                elif offloaded >= total:
                    status = "ALL"
                else:
                    status = "PARTIAL"
            except:
                status = "-"
        elif sgs.isdigit() and int(sgs) == 0:
            status = "NONE"
        else:
            status = "-"

        if nodes != "Not detected":
            offload_text = f"{status} - {sgs} subgraph(s) [{nodes} nodes]"
        else:
            offload_text = f"{status} - {sgs} subgraph(s)"

        offload_div = html.div(html.p(offload_text, style="margin: 0;"))
        cells.insert(2, html.td(offload_div))

    # Add performance metrics
    if hasattr(report, 'perf_metrics') and report.perf_metrics:
        pm = report.perf_metrics
        perf_div = html.div()
        if pm.get('wall_time'):
            perf_div.append(html.p(f"wall_time: {pm['wall_time']}", style="margin: 0; color: gray;"))
        if pm.get('total_time'):
            perf_div.append(html.p(f"total_time: {pm['total_time']}", style="margin: 0;"))
        if pm.get('core_time'):
            perf_div.append(html.p(f"core_time: {pm['core_time']}", style="margin: 0;"))
        if pm.get('subgraph_time') and pm['subgraph_time'][1]:
            sg_label, sg_val = pm['subgraph_time']
            perf_div.append(html.p(f"{sg_label}: {sg_val}", style="margin: 0;"))
        if pm.get('read_total'):
            perf_div.append(html.p(f"read_total: {pm['read_total']}", style="margin: 0;"))
        if pm.get('write_total'):
            perf_div.append(html.p(f"write_total: {pm['write_total']}", style="margin: 0;"))
        if pm.get('ddr_total'):
            perf_div.append(html.p(f"ddr_total: {pm['ddr_total']}", style="margin: 0;"))
        cells.insert(3, html.td(perf_div))
    else:
        cells.insert(3, html.td("-"))

    # Add output metrics if available
    if(hasattr(report,'nmse') or hasattr(report,'mse') or hasattr(report,'max_delta') or hasattr(report,'post_proc_metrics')):
        metrics = []
        if(hasattr(report,'nmse') and report.nmse != '-' and report.nmse != 'None'):
            metrics.append(f"MAX NMSE: {report.nmse}")
        if(hasattr(report,'mse') and report.mse != '-' and report.mse != 'None'):
            metrics.append(f"MAX MSE: {report.mse}")
        if(hasattr(report,'max_delta') and report.max_delta != '-' and report.max_delta != 'None'):
            metrics.append(f"MAX DELTA: {report.max_delta}")

        if not metrics and hasattr(report, 'post_proc_metrics') and report.post_proc_metrics != '-':
            metrics.append(f"{report.post_proc_metrics}")

        if metrics:
            metrics_div = html.div()
            for i, metric in enumerate(metrics):
                metrics_div.append(html.p(metric, style="margin: 0;"))
            cells.insert(4, html.td(metrics_div))
        else:
            cells.insert(4, html.td("-"))

    # Add plot image if available
    if hasattr(report, 'plot_data') and report.plot_data:
        img_html = html.div(
            html.img(src=f"data:image/png;base64,{report.plot_data}",
                    style="max-width:250px; cursor:pointer; margin:0; padding:0;",
                    onclick="window.open(this.src)"),
            style="text-align:center; margin:0; padding:0;"
        )
        cells.insert(5, html.td(img_html, style="text-align:center; margin:0; padding:0;"))
    else:
        cells.insert(5, html.td("-"))
