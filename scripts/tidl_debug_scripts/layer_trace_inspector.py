import os
import glob
import logging
import argparse
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger("visualization_logger")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traceONNX", default="<path to onnx trace dump>")
    parser.add_argument("--traceTIDL", default="<path to tidl trace dump>")
    parser.add_argument("--traceInfo", default="<path to layerinfo.txt>")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable debug traces")
    args = parser.parse_args()
    return args

class VisualizationUtils:
    def sideBySidePlots(self, goldenBuffer, quantBuffer):
        fig,a =  plt.subplots(2,1)

        a[0].plot(goldenBuffer.flatten(), quantBuffer.flatten(), 'r.')
        a[0].set_title("ONNX vs TIDL")
        a[0].set_xlabel('ONNX Output')
        a[0].set_ylabel('TIDL Output')
        a[0].set_ylim([goldenBuffer.min(), goldenBuffer.max()])

        a[1].plot(goldenBuffer.flatten(), 'bs')
        a[1].plot(quantBuffer.flatten(), 'c.')
        a[1].set_title('ONNX vs TIDL')
        a[1].set_xlabel('ONNX Output')
        a[1].set_ylabel('TIDL Output')

        fig.tight_layout()
        st.pyplot(fig)

    def plotHistogram(self, fig, buffer, title, plotIdx):
        ax = fig.add_subplot(3, 1, plotIdx)
        ax.hist(buffer, bins=1024)
        ax.set_title(title)

    def plotCombinedHistograms(self, goldenBuffer, quantBuffer):
        fig =  plt.figure()
        flatGolden = goldenBuffer.flatten()
        flatQuant  = quantBuffer.flatten()
        self.plotHistogram(fig, flatGolden, " ONNX Trace:", 1)
        self.plotHistogram(fig, flatQuant, " TIDL Trace:", 2)
        fig.tight_layout()
        st.pyplot(fig)

class LayerTraceInspector(VisualizationUtils):
    def __init__(self, onnx_trace_folder: str, tidl_trace_folder: str, layer_info_path: str):
        self.onnx_trace_folder = onnx_trace_folder
        self.tidl_trace_folder = tidl_trace_folder
        self.layer_info_path = layer_info_path
        self.tidl_onnx_trace_mapping = {}
        self.getTraces()

    def getTraces(self):
        entries = [line.strip().split(" ") for line in open(self.layer_info_path)]
        onnx_entries = os.listdir(self.onnx_trace_folder)
        for entry in entries:
            if entry[0] != entry[1]:
                continue
            tidl_data_id = entry[0]
            onnx_layer_name = entry[-1]
            onnx_layer_id = onnx_layer_name.replace("/", "_")

            _tidl_data_id = tidl_data_id
            while len(_tidl_data_id) < 4:
                _tidl_data_id = "0" + _tidl_data_id
            
            _tidl_trace_path = os.path.join(self.tidl_trace_folder, f"tidl_trace_subgraph_0_{_tidl_data_id}*_float.bin")

            onnx_trace_path = list(filter(lambda path: onnx_layer_id in path, onnx_entries))
            tidl_trace_path = glob.glob(_tidl_trace_path)

            if len(onnx_trace_path) > 0 and len(tidl_trace_path) > 0:
                onnx_trace_path = os.path.join(self.onnx_trace_folder, onnx_trace_path[0])
                tidl_trace_path = tidl_trace_path[0]
                self.tidl_onnx_trace_mapping[tidl_data_id] = [os.path.basename(onnx_trace_path), os.path.basename(tidl_trace_path)] 
            else:
                logger.debug(f"Traces Not found for outdataId: {_tidl_data_id}")

    def singlePlot(self):
        start_idx = 0
        end_idx = 0
        with st.sidebar:
            option = st.selectbox("Pick TIDL Trace to compare", self.tidl_onnx_trace_mapping.keys())
            st.write(f"TIDL Trace {self.tidl_onnx_trace_mapping[option][1]}")
            st.write(f"ONNX Trace {self.tidl_onnx_trace_mapping[option][0]}")
            start_idx = st.text_input("Start Idx","NA")
            end_idx = st.text_input("End Idx","NA")
            try:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
            except:
                start_idx = "NA"
                end_idx = "NA"

        genGraph = st.button("Generate ", type="primary")
        if genGraph:
            optionPath = self.tidl_onnx_trace_mapping[option][1]
            optionPathFloat = self.tidl_onnx_trace_mapping[option][0]
            genGraph = False

            goldenBuffer = np.fromfile(os.path.join(self.onnx_trace_folder, optionPathFloat), dtype=np.float32)
            quantBuffer = np.fromfile(os.path.join(self.tidl_trace_folder, optionPath), dtype=np.float32)

            logger.info(f"Constraining between {start_idx} : {end_idx}")
            if start_idx != "NA" and end_idx != "NA":
                goldenBuffer = goldenBuffer[start_idx:end_idx]
                quantBuffer = quantBuffer[start_idx:end_idx]

            delta = goldenBuffer - quantBuffer
            delta = np.absolute(delta).flatten()
            delta_max = np.argmax(delta)

            if st.sidebar.button('Generate Histogram plots'):
                with st.spinner("Generating histograms!"):
                    self.plotCombinedHistograms(goldenBuffer, quantBuffer)
            else:
                with st.spinner("Generating scatter plots!"):
                    self.sideBySidePlots(goldenBuffer, quantBuffer)
                    self.plotCombinedHistograms(goldenBuffer, quantBuffer)

            with st.sidebar:
                st.write(f'MAX ERROR:')
                st.write(np.max(delta))
                st.write(f'AVG ERROR:')
                st.write(np.mean(delta))
                st.write(f"Max delta {delta[delta_max]} at")
                st.write(delta_max)
                st.write(f'TIDL Buf={quantBuffer[delta_max]}')
                st.write(f'ONNX Buf={goldenBuffer[delta_max]}')
    
    def network_error_summary(self):
        dropdown_list = list(self.tidl_onnx_trace_mapping.keys())
        numLayers = len(dropdown_list)
        mae_dict = {}
        max_dict = {}
        mae_abs_dict = {}
        for idx in range(numLayers):
            tidl = os.path.join(self.tidl_trace_folder, self.tidl_onnx_trace_mapping[dropdown_list[idx]][1])
            golden = os.path.join(self.onnx_trace_folder, self.tidl_onnx_trace_mapping[dropdown_list[idx]][0])
            goldenBuffer = np.fromfile(golden,dtype=np.float32).flatten()
            tidlBuffer = np.fromfile(tidl,dtype=np.float32).flatten()
            delta= []
            try:
                delta = goldenBuffer - tidlBuffer
            except:
                delta = np.zeros_like(tidlBuffer)
            scale = np.mean(np.absolute(goldenBuffer))
            scale = np.abs(scale)
            abs_delta = np.absolute(delta)
            max = np.max(abs_delta)
            mae = np.mean(abs_delta)/scale
            mae_dict[dropdown_list[idx]] = mae
            max_dict[dropdown_list[idx]] = max
            mae_abs_dict[dropdown_list[idx]] = np.mean(np.absolute(delta))

        #Plot mae_dict and max_dict:
        fig,a =  plt.subplots(3,1)
        a[0].plot(dropdown_list,list(mae_dict.values()))
        a[0].set_title("MAE")
        a[0].set_xlabel('LAYERS')
        a[0].set_ylabel('ERROR (REL MAE)')
        a[1].plot(dropdown_list, list(max_dict.values()))
        a[1].set_xlabel('LAYERS')
        a[1].set_ylabel('ERROR (MAX)')
        a[2].plot(dropdown_list,list(mae_abs_dict.values()))
        a[2].set_xlabel('LAYERS')
        a[2].set_ylabel('ERROR (ABS MAE)')


        df1 = pd.DataFrame(dict(
            layerId=dropdown_list,
            error=list(mae_dict.values()),
        ))
        st.plotly_chart(px.line(df1, x='layerId', y='error', title="Error (Rel MAE)"))

        df2 = pd.DataFrame(dict(
            x=dropdown_list,
            y=list(max_dict.values()),
        ))
        st.plotly_chart(px.line(df2, x='x', y='y', title="Error (MAX)"))

        df3 = pd.DataFrame(dict(
            x=dropdown_list,
            y=list(mae_abs_dict.values()),
        ))
        st.plotly_chart(px.line(df3, x='x', y='y', title="Error (ABS MAE)"))

    def multiplot(self):
        drop_down_list = list(self.tidl_onnx_trace_mapping.keys())

        with st.sidebar:
            options = st.multiselect("Pick set of TIDL Traces to compare",drop_down_list)
            genGraph = st.button("Generate MultiPlot ", type="primary")
        for option in options:            
            if True:
                with st.expander(f'Plots for {option}'):
                    col1, col2 = st.columns(2)
                    optionPath = self.tidl_onnx_trace_mapping[option][1]
                    optionPathFloat = self.tidl_onnx_trace_mapping[option][0]
                    genGraph = False
                    goldenBuffer = np.fromfile(self.onnx_trace_folder + os.path.sep + optionPathFloat, dtype=np.float32)
                    quantBuffer = np.fromfile(self.tidl_trace_folder + os.path.sep + optionPath, dtype=np.float32)
                    if(len(goldenBuffer) != len(quantBuffer)):
                        st.write(f"Unable to compare {option}")
                        st.write(f"{self.tidl_onnx_trace_mapping[option][1]} & {self.tidl_onnx_trace_mapping[option][0]}")
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
                        # with st.sidebar:
                        st.write(f'MAX, AVG & Relative Average ERR:')
                        st.write(np.max(delta))
                        st.write(delta_val)
                        st.write(delta_val/scale)
                        st.write(f"Max delta {delta[delta_max]} at")
                        st.write(delta_max)
                        st.write(f'TIDL Buf={quantBuffer[delta_max]}')
                        st.write(f'ONNX Buf={goldenBuffer[delta_max]}') 
                        st.divider()

if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    layer_trace_inspector = LayerTraceInspector(args.traceONNX, args.traceTIDL, args.traceInfo)

    pg = st.navigation([
        st.Page(layer_trace_inspector.singlePlot, title="SinglePlot"),
        st.Page(layer_trace_inspector.network_error_summary, title="Network Error Summary"),
        st.Page(layer_trace_inspector.multiplot, title="Mulit Plot"),
    ])
    pg.run()