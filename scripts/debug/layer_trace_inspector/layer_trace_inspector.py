import os
import glob
import logging
import argparse
import pathlib
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

logging.basicConfig(format="[%(levelname)s] %(message)s")
logger = logging.getLogger("visualization_logger")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--traceRef", type=str, help="<path to reference output binaries>", default = None)
    parser.add_argument("--traceTest", type=str, help="<path to test output binaries>", default = None)
    parser.add_argument("--traceInfo", type=str, help="<path to layer info file>(Optional)", default = None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable debug traces")
    args = parser.parse_args()
    return args


class VisualizationUtils:
    def sideBySidePlots(self, goldenBuffer: np.ndarray, quantBuffer: np.ndarray):
        """
        Plots side-by-side comparisons of two data buffers: a golden reference buffer
        (REF output) and a quantized buffer (TEST output).
        """
        fig, a = plt.subplots(2, 1)
        min_val = goldenBuffer.min()
        max_val = goldenBuffer.max()

        a[0].plot(goldenBuffer.flatten(), quantBuffer.flatten(), "r.")
        a[0].plot([min_val, max_val], [min_val, max_val], "b-", alpha=0.7)
        a[0].set_title("REF vs TEST")
        a[0].set_xlabel("REF Output")
        a[0].set_ylabel("TEST Output")
        a[0].set_ylim([min_val, max_val])

        a[1].plot(goldenBuffer.flatten(), "bs")
        a[1].plot(quantBuffer.flatten(), "c.")
        a[1].set_title("")
        a[1].set_xlabel("Index")
        a[1].set_ylabel("Output")
        fig.tight_layout()
        st.pyplot(fig)

    def plotHistogram(self, fig, buffer: np.ndarray, title: str, plotIdx: int):
        """
        Plots a histogram of the given data buffer on a specified subplot.
        """
        ax = fig.add_subplot(3, 1, plotIdx)
        ax.hist(buffer, bins=1024)
        ax.set_title(title)

    def plotCombinedHistograms(self, goldenBuffer: np.ndarray, quantBuffer: np.ndarray):
        """
        Plots combined histograms for two input buffers: a golden reference buffer(REF Output) and a quantized buffer(TEST Output).
        """
        try:
            fig = plt.figure()
            fig.suptitle("Value Distribution Histograms", fontsize=14)
            flatGolden = goldenBuffer.flatten()
            flatQuant = quantBuffer.flatten()
            self.plotHistogram(fig, flatGolden, " REF Trace:", 1)
            self.plotHistogram(fig, flatQuant, " TEST Trace:", 2)
            fig.tight_layout()
            st.pyplot(fig)
        except:
            pass

class LayerTraceInspector(VisualizationUtils):
    """
    LayerTraceInspector is a utility class for visualizing and analyzing output binaries.
    It provides methods to map traces between the two frameworks, generate plots for 
    individual or multiple layers, and summarize network-level errors.
    """

    def __init__(
        self, ref_trace_folder: str, test_trace_folder: str, layer_info_path: str = None
    ):
        self.ref_trace_folder = ref_trace_folder
        self.test_trace_folder = test_trace_folder
        self.layer_info_path = layer_info_path
        self.trace_mapping = None
        self.ref_file_list = [file for file in os.listdir(self.ref_trace_folder) if (pathlib.Path(file).suffix == ".bin")]
        self.test_file_list = [file for file in os.listdir(self.test_trace_folder) if (pathlib.Path(file).suffix == ".bin")]

        if self.layer_info_path != None:
            self.trace_mapping = self.getTraces()

    def getTraces(self) -> dict[str, list[str]]:
        """
        Computes the mapping between RED and TEST traces from layer_info_path
        """
        trace_mapping = {}
        entries = [line.strip().split(" ") for line in open(self.layer_info_path)]
        for entry in entries:
            if entry[0] != entry[1]:
                continue
            test_data_id = entry[0]
            ref_layer_name = entry[-1]
            ref_layer_id = ref_layer_name.replace("/", "_")

            _test_data_id = test_data_id
            while len(_test_data_id) < 4:
                _test_data_id = "0" + _test_data_id

            _test_trace_path = os.path.join(
                self.test_trace_folder,
                f"tidl_trace_subgraph_0_{_test_data_id}*_float.bin",
            )

            ref_trace_path = list(
                filter(lambda path: ref_layer_id in path, self.ref_file_list)
            )
            test_trace_path = glob.glob(_test_trace_path)

            if len(ref_trace_path) > 0 and len(test_trace_path) > 0:
                ref_trace_path = os.path.join(
                    self.ref_trace_folder, ref_trace_path[0]
                )
                test_trace_path = test_trace_path[0]
                trace_mapping[test_data_id] = [
                    os.path.basename(ref_trace_path),
                    os.path.basename(test_trace_path),
                ]
            else:
                logger.debug(f"Traces Not found for outdataId: {_test_data_id}")
        return trace_mapping
    
    def singlePlotNoInfo(self):
        """
        Generate plots for a single layer withput layer info
        """
        dataIdMapRef = {}
        dataIdMapTest = {}

        for file in self.ref_file_list:
            dataIdMapRef[file] = file

        for file in self.test_file_list:
            dataIdMapTest[file] = file

        start_idx = 0
        end_idx = 0
        with st.sidebar:
            optionTest = st.selectbox("Pick test trace to compare against",dataIdMapTest.keys())
            st.write(f"Test Trace: {optionTest}")
            optionRef = st.selectbox("Pick ref trace to compare",dataIdMapRef.keys())
            st.write(f"Ref Trace: {optionRef}")

            st.divider()
            start_idx = st.text_input("Start Idx","NA")
            end_idx = st.text_input("End Idx","NA")
            try:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
            except:
                start_idx = "NA"
                end_idx = "NA"
            st.divider()

        genGraph = st.button("Generate ", type="primary")
        if genGraph:
            optionPathTest = dataIdMapTest[optionTest]
            optionPathRef = dataIdMapRef[optionRef]
            genGraph = False

            goldenBuffer = np.fromfile(
                os.path.join(self.ref_trace_folder, optionPathRef), dtype=np.float32
            )
            quantBuffer = np.fromfile(
                os.path.join(self.test_trace_folder, optionPathTest), dtype=np.float32
            )

            
            logger.info(f"Constraining between {start_idx} : {end_idx}")
            if start_idx != "NA" and end_idx != "NA":
                goldenBuffer = goldenBuffer[start_idx:end_idx]
                quantBuffer = quantBuffer[start_idx:end_idx]
            
            # Check if golden trace and reference trace match in length
            if len(goldenBuffer) != len(quantBuffer):
                st.warning(f"⚠️ REF TRACE and TEST TRACE DO NOT MATCH IN LENGTH! REF:{len(goldenBuffer)}, TEST:{len(quantBuffer)}")
                logger.warning(f"Trace length mismatch: Ref={len(goldenBuffer)}, Test={len(quantBuffer)}")
                return

            delta = goldenBuffer - quantBuffer
            delta = np.absolute(delta).flatten()
            delta_max = np.argmax(delta)

            with st.spinner("Generating scatter plots!"):
                self.sideBySidePlots(goldenBuffer, quantBuffer)
                self.plotCombinedHistograms(goldenBuffer, quantBuffer)

            with st.sidebar:
                st.write(f"MAX ERROR:", np.max(delta))
                st.write(f"AVG ERROR:", np.mean(delta))
                st.write(f"MAX DELTA ", delta[delta_max], " at", delta_max)
                st.write(f"TEST:", quantBuffer[delta_max])
                st.write(f"REF:", goldenBuffer[delta_max])
                st.divider()

                # Add collapsible information section about plots
                with st.sidebar.expander("ℹ️ Plot Information"):
                    st.markdown("""
                    ### Available Plots

                    **Scatter Plot**
                    - Shows correlation between REF and TEST values
                    - X-axis: REF Output values
                    - Y-axis: TEST Output values
                    - Red Dots: REF/TEST pairs, Blue Line: 45° equality line
                    - Good match when points lie close to the blue line
                    - EX: If Ref Output = [2,6,8] and Test Output = [2.4,7,8] plots will be at (2,2.4), (6,7) and (8,8)

                    **Time Series Plot**
                    - Shows how values change over the sequence
                    - Blue Squares: REF values, Cyan Dots: TEST values
                    - Helps identify where differences occur
                    - Good match when points overlap

                    **Histogram Plots**
                    - Shows distribution of values in each buffer
                    - Upper: REF values distribution
                    - Lower: TEST values distribution
                    - Comparing shapes helps identify distribution differences
                    """)

    def singlePlot(self):
        """
        Generate plots for a single layer
        """
        start_idx = 0
        end_idx = 0
        with st.sidebar:
            option = st.selectbox(
                "Pick Test Trace to compare", self.trace_mapping.keys()
            )
            st.write(f"Test Trace: {self.trace_mapping[option][1]}")
            st.write(f"Ref Trace: {self.trace_mapping[option][0]}")

            st.divider()
            start_idx = st.text_input("Start Idx", "NA")
            end_idx = st.text_input("End Idx", "NA")
            try:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
            except:
                start_idx = "NA"
                end_idx = "NA"
            st.divider()

        genGraph = st.button("Generate ", type="primary")
        if genGraph:
            optionPathTest = self.trace_mapping[option][1]
            optionPathRef = self.trace_mapping[option][0]
            genGraph = False

            goldenBuffer = np.fromfile(
                os.path.join(self.ref_trace_folder, optionPathRef), dtype=np.float32
            )
            quantBuffer = np.fromfile(
                os.path.join(self.test_trace_folder, optionPathTest), dtype=np.float32
            )

            logger.info(f"Constraining between {start_idx} : {end_idx}")
            if start_idx != "NA" and end_idx != "NA":
                goldenBuffer = goldenBuffer[start_idx:end_idx]
                quantBuffer = quantBuffer[start_idx:end_idx]
            
            # Check if golden trace and reference trace match in length
            if len(goldenBuffer) != len(quantBuffer):
                st.warning(f"⚠️ REF TRACE and TEST TRACE DO NOT MATCH IN LENGTH! REF:{len(goldenBuffer)}, TEST:{len(quantBuffer)}")
                logger.warning(f"Trace length mismatch: Ref={len(goldenBuffer)}, Test={len(quantBuffer)}")
                return

            delta = goldenBuffer - quantBuffer
            delta = np.absolute(delta).flatten()
            delta_max = np.argmax(delta)

            with st.spinner("Generating scatter plots!"):
                self.sideBySidePlots(goldenBuffer, quantBuffer)
                self.plotCombinedHistograms(goldenBuffer, quantBuffer)

            with st.sidebar:
                st.write(f"MAX ERROR:", np.max(delta))
                st.write(f"AVG ERROR:", np.mean(delta))
                st.write(f"MAX DELTA ", delta[delta_max], " at", delta_max)
                st.write(f"TEST:", quantBuffer[delta_max])
                st.write(f"REF:", goldenBuffer[delta_max])
                st.divider()

                # Add collapsible information section about plots
                with st.sidebar.expander("ℹ️ Plot Information"):
                    st.markdown("""
                    ### Available Plots

                    **Scatter Plot**
                    - Shows correlation between REF and TEST values
                    - X-axis: REF Output values
                    - Y-axis: TEST Output values
                    - Red Dots: REF/TEST pairs, Blue Line: 45° equality line
                    - Good match when points lie close to the blue line
                    - EX: If Ref Output = [2,6,8] and Test Output = [2.4,7,8] plots will be at (2,2.4), (6,7) and (8,8)

                    **Time Series Plot**
                    - Shows how values change over the sequence
                    - Blue Squares: REF values, Cyan Dots: TEST values
                    - Helps identify where differences occur
                    - Good match when points overlap

                    **Histogram Plots**
                    - Shows distribution of values in each buffer
                    - Upper: REF values distribution
                    - Lower: TEST values distribution
                    - Comparing shapes helps identify distribution differences
                    """)

    def networkErrorSummary(self):
        """
        Generate error summary for the entire network
        """ 
        dropdown_list = list(self.trace_mapping.keys())
        numLayers = len(dropdown_list)
        mae_dict = {}
        max_dict = {}
        mae_abs_dict = {}
        for idx in range(numLayers):
            tidl = os.path.join(
                self.test_trace_folder,
                self.trace_mapping[dropdown_list[idx]][1],
            )
            golden = os.path.join(
                self.ref_trace_folder,
                self.trace_mapping[dropdown_list[idx]][0],
            )
            goldenBuffer = np.fromfile(golden, dtype=np.float32).flatten()
            tidlBuffer = np.fromfile(tidl, dtype=np.float32).flatten()
            delta = []
            try:
                delta = goldenBuffer - tidlBuffer
            except:
                delta = np.zeros_like(tidlBuffer)
            scale = np.mean(np.absolute(goldenBuffer))
            scale = np.abs(scale)
            abs_delta = np.absolute(delta)
            max = np.max(abs_delta)
            mae = np.mean(abs_delta) / scale
            mae_dict[dropdown_list[idx]] = mae
            max_dict[dropdown_list[idx]] = max
            mae_abs_dict[dropdown_list[idx]] = np.mean(np.absolute(delta))

        # Plot mae_dict and max_dict:
        fig, a = plt.subplots(3, 1)
        a[0].plot(dropdown_list, list(mae_dict.values()))
        a[0].set_title("MAE")
        a[0].set_xlabel("LAYERS")
        a[0].set_ylabel("ERROR (REL MAE)")
        a[1].plot(dropdown_list, list(max_dict.values()))
        a[1].set_xlabel("LAYERS")
        a[1].set_ylabel("ERROR (MAX)")
        a[2].plot(dropdown_list, list(mae_abs_dict.values()))
        a[2].set_xlabel("LAYERS")
        a[2].set_ylabel("ERROR (ABS MAE)")

        df1 = pd.DataFrame(
            dict(
                layerId=dropdown_list,
                error=list(mae_dict.values()),
            )
        )
        st.plotly_chart(px.line(df1, x="layerId", y="error", title="Error (Rel MAE)"))

        df2 = pd.DataFrame(
            dict(
                x=dropdown_list,
                y=list(max_dict.values()),
            )
        )
        st.plotly_chart(px.line(df2, x="x", y="y", title="Error (MAX)"))
        df3 = pd.DataFrame(
            dict(
                x=dropdown_list,
                y=list(mae_abs_dict.values()),
            )
        )
        st.plotly_chart(px.line(df3, x="x", y="y", title="Error (ABS MAE)"))

    def multiplot(self):
        """
        Generate plots for multiple layers
        """
        drop_down_list = list(self.trace_mapping.keys())

        with st.sidebar:
            options = st.multiselect(
                "Pick set of TEST Traces to compare", drop_down_list
            )
            genGraph = st.button("Generate MultiPlot ", type="primary")
        for option in options:
            if True:
                with st.expander(f"Plots for {option}"):
                    col1, col2 = st.columns(2)
                    optionPathTest = self.trace_mapping[option][1]
                    optionPathRef = self.trace_mapping[option][0]
                    genGraph = False
                    goldenBuffer = np.fromfile(
                        self.ref_trace_folder + os.path.sep + optionPathRef,
                        dtype=np.float32,
                    )
                    quantBuffer = np.fromfile(
                        self.test_trace_folder + os.path.sep + optionPathTest,
                        dtype=np.float32,
                    )
                    if len(goldenBuffer) != len(quantBuffer):
                        st.warning(f"⚠️ REF TRACE and TEST TRACE DO NOT MATCH IN LENGTH! REF:{len(goldenBuffer)}, TEST:{len(quantBuffer)}")
                        st.write(f"Unable to compare {option}: {self.trace_mapping[option][1]} & {self.trace_mapping[option][0]}")
                        logger.warning(f"Trace length mismatch for {option}: Golden={len(goldenBuffer)}, Ref={len(quantBuffer)}")
                        continue

                    delta = goldenBuffer - quantBuffer
                    delta = np.absolute(delta).flatten()
                    delta_max = np.argmax(delta)
                    scale = np.mean(np.absolute(goldenBuffer))
                    delta_val = np.mean((delta))
                    with col1:
                        with st.spinner("Generating scatter plots!"):
                            self.sideBySidePlots(goldenBuffer, quantBuffer)
                            self.plotCombinedHistograms(goldenBuffer, quantBuffer)
                    with col2:
                        st.divider()
                        st.write(f"MAX ERROR:", np.max(delta))
                        st.write(f"AVG ERROR:", delta_val)
                        st.write(f"RELATIVE AVG ERROR:", (delta_val / scale))
                        st.write(f"MAX DELTA ", delta[delta_max], " at", delta_max)
                        st.write(f"TEST:", quantBuffer[delta_max])
                        st.write(f"REF:", goldenBuffer[delta_max])
                        st.divider()



if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if traceRef and traceTest are directories
    if args.traceRef is None or not os.path.isdir(args.traceRef):
        logger.error(f"traceRef '{args.traceRef}' is not a valid directory")
        exit(1)
    
    if args.traceTest is None or not os.path.isdir(args.traceTest):
        logger.error(f"traceTest '{args.traceTest}' is not a valid directory")
        exit(1)
    
    if args.traceInfo is not None and not os.path.isfile(args.traceInfo):
        logger.error(f"traceInfo '{args.traceInfo}' is not a valid file")
        exit(1)

    layer_trace_inspector = LayerTraceInspector(
        args.traceRef, args.traceTest, args.traceInfo
    )

    if args.traceInfo is not None:
        pg = st.navigation(
            [
                st.Page(layer_trace_inspector.singlePlot, title="SinglePlot"),
                st.Page(
                    layer_trace_inspector.networkErrorSummary,
                    title="Network Error Summary",
                ),
                st.Page(layer_trace_inspector.multiplot, title="Mulit Plot"),
            ]
        )
    else:
        pg = st.navigation(
            [
                st.Page(layer_trace_inspector.singlePlotNoInfo, title="SinglePlot"),
            ]
        )
    pg.run()
