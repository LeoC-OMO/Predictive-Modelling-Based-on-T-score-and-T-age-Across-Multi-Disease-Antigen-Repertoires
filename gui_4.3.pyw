import tkinter as tk
from tkinter import filedialog as fd, messagebox
from tkinter.messagebox import showinfo
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pipeline_functions as pf
from typing import List
from matplotlib.backends.backend_pdf import PdfPages
import os
import append_header_to_csv as ap
import extract_7k as ex
import msdos_to_csv as mtc
import threading
from pathlib import Path

# ====================== GLOBAL VARIABLES ======================
top_n = 3500
threshold_mode_used = "topN"
window = None
result_tree = None
selected_folder = ""
three_group_model = None
binary_disease_model = None
binary_shealthy_model = None
binary_healthy_model = None
current_sample_df = None
current_model = None
all_prediction_results = None   # list of (filename, predicted, score, sample_df)
analysis_cache = None   # Will store the 4 models' results
final_prediction_cache = None  # stores final ensemble result
current_sample_full_result = None
btn_choose = None
btn_analyze = None
progress_bar = None
overlay_frame = None
initialization_label = None
extra_props_df = None # hold extra info about features
# ====================== EXPLANATION FUNCTIONS ======================



def calculate_health_score(multi_scores: dict, final_scores: dict) -> float:
    """
    Improved health score that combines:
    - Three-group model scores
    - Final ensemble scores
    """
    # Get scores
    h_multi  = multi_scores.get('healthy', 0.0)
    sh_multi = multi_scores.get('subhealthy', 0.0)
    d_multi  = multi_scores.get('disease', 0.0)
    
    h_final  = final_scores.get('healthy', 0.0)
    sh_final = final_scores.get('subhealthy', 0.0)
    d_final  = final_scores.get('disease', 0.0)

    # Weighted combination (give more weight to final ensemble)
    h  = 0.4 * h_multi  + 0.6 * h_final
    sh = 0.4 * sh_multi + 0.6 * sh_final
    d  = 0.4 * d_multi  + 0.6 * d_final

    # Softmax with temperature to make differences more visible
    scores = np.array([h, sh, d])
    exp_scores = np.exp(scores / 0.9)          # temperature = 0.8 makes it sharper
    probabilities = exp_scores / np.sum(exp_scores)
    
    health_score = 100.0 * probabilities[0]
    
    # Bonus if healthy is clearly the highest
    # if h > sh and h > d and h > 0:
    if h > sh + 0.05 and h > d + 0.05:
        health_score = min(100.0, health_score * 1.25)
    
    # Final gentle scaling to push good cases higher
    health_score = health_score ** 0.92
    
    # return round(health_score, 1)
    return round(max(0.0, min(100.0, health_score)), 1)

def calculate_health_score_spectrum(multi_scores: dict, final_scores: dict) -> float:
    """
    Continuous Health Index using a weighted spectrum and logistic stretching.
    This allows for realistic overlap between healthy, sub-healthy, and disease.
    """
# 1. Extract raw values
    # Example input: h=-0.036, sh=-0.048, d=-0.045
    h_raw  = 0.40 * multi_scores.get('healthy', 0.0)    + 0.60 * final_scores.get('healthy', 0.0)
    sh_raw = 0.35 * multi_scores.get('subhealthy', 0.0) + 0.65 * final_scores.get('subhealthy', 0.0)
    d_raw  = 0.35 * multi_scores.get('disease', 0.0)    + 0.65 * final_scores.get('disease', 0.0)
    
    print(f"DEBUG - Raw Scores: H={h_raw:.5f}, SH={sh_raw:.5f}, D={d_raw:.5f}")

    # 2. Convert raw negatives to probabilities using Softmax
    logits = np.array([h_raw, sh_raw, d_raw])
    amplified = logits * 19.0
    exp_logits = np.exp(amplified)
    probs = exp_logits / np.sum(exp_logits)
    
    h_p, sh_p, d_p = probs[0], probs[1], probs[2]
    
    print(f"DEBUG - Probabilities: H={h_p:.4f}, SH={sh_p:.4f}, D={d_p:.4f} | Sum={h_p+sh_p+d_p:.4f}")

    # 3. Map to 0-100 Spectrum
    # Healthy = 100, Sub-healthy = 42, Disease = 8
    health_score = (h_p * 100.0) + (sh_p * 42.0) + (d_p * 8.0)
    diff_h = h_p - max(sh_p, d_p)

    # 4. Stretch the results 
    if diff_h > 0.08: # test how healthy sample is using difference in scores. Different tiers gets different bonuses in scores
        health_score = min(99.0, health_score * 1.75)
        print("DEBUG - VERY STRONG healthy bonus")
    elif diff_h > 0.04:
        health_score = min(98.5, health_score * 1.58)
        print("DEBUG - Strong healthy bonus")
    elif diff_h > 0.018:
        health_score = min(97.0, health_score * 1.35)
        print("DEBUG - Moderate healthy bonus")
    elif diff_h > 0.005:
        health_score = min(94.0, health_score * 1.18)
        print("DEBUG - Small healthy edge bonus")

    if d_p > h_p + 0.07 and d_p > sh_p + 0.07: # penalty given to really sick cases
        health_score = max(5.0, health_score * 0.55)
        print("DEBUG - Disease penalty applied")

    final_score = round(max(5.0, min(99.0, health_score)), 1)
    print(f"DEBUG - Final Health Score: {final_score}%\n")
    return final_score

def clip(n, smallest, largest):
    return max(smallest, min(n, largest))

def explain_prediction(sample_df: pd.DataFrame, model: dict, top_n: int = 20):
    """Calculate contribution of each triplet."""
    sample_feats = pf.threshold_to_features(sample_df) # return top n features 
    
    if model == "three_group_model":
        print("3 group now")
        _, scores = pf.tscore_predict_group(sample_df, model)
    else:
        _, scores = pf.tscore_predict_binary(sample_df, model)
    predicted_group = max(scores, key=scores.get)
    
    contributions = []
    
    if not model.get('features'):
        print("DEBUG: This model has no features! Check your filters in initialization().")
        # Return an empty DF with the expected columns so the GUI doesn't crash
        return pd.DataFrame(columns=['triplet', 'abundance', 'contribution', 'direction', 'abs_contrib']), predicted_group, scores
    
    for f in model['features']:
        # print(len(model))
        mu = model['per_group'][predicted_group][f]['mean']
        sd = model['per_group'][predicted_group][f]['std']
        x = sample_feats.get(f, 0.0)
        w = model['weights'].get(f, 0.0)
        
        z = (x - mu) / sd if sd > 0 else 0.0 # calculate z-score of triplet (the relative change)
        contrib = w * z # multiply z-score with weight to get amount contributed to final score
        
        contributions.append({
            'triplet': f,
            'abundance': round(x, 6),
            'contribution': round(contrib, 6),
            'direction': '↑ Toward' if contrib > 0 else '↓ Against'
        })
    
    contrib_df = pd.DataFrame(contributions)
    contrib_df['abs_contrib'] = contrib_df['contribution'].abs()
    contrib_df = contrib_df.sort_values(by='abs_contrib', ascending=False).head(top_n)
    
    return contrib_df, predicted_group, scores


def show_explanation():
    global analysis_cache, final_prediction_cache
    if current_sample_df is None:
        showinfo("Error", "Please select a sample from the results table first!")
        return
    
        
    # 1. Create the Popup Window
    popup = tk.Toplevel(window)
    popup.title("Multi-Model Feature Analysis")
    popup.geometry("1300x900")
    
    loading_frame = tk.Frame(popup, bg="#f0f0f0")
    loading_frame.pack(expand=True, fill="both")

    loading_label = tk.Label(loading_frame, text="Calculating detailed analysis...\nPlease wait", 
                             font=("Helvetica", 14), bg="#f0f0f0")
    loading_label.pack(pady=40)

    progress = ttk.Progressbar(loading_frame, mode='indeterminate', length=400)
    progress.pack(pady=20)
    progress.start(10)

    # Force GUI update
    popup.update()
    
    
    # List of models and their display names
    analysis_models = [
        (three_group_model, "Ensemble (3-Group)"),
        (binary_healthy_model, "Binary: Healthy"),
        (binary_shealthy_model, "Binary: Sub-Healthy"),
        (binary_disease_model, "Binary: Disease")
    ]
    
    try:
        multi_group, multi_scores = pf.tscore_predict_group(current_sample_df, three_group_model)
        bin_h = pf.tscore_predict_binary(current_sample_df, binary_healthy_model)[1]
        bin_sh = pf.tscore_predict_binary(current_sample_df, binary_shealthy_model)[1]
        bin_d = pf.tscore_predict_binary(current_sample_df, binary_disease_model)[1]

        final_scores = pf.compute_final_scores(multi_scores, [bin_h, bin_sh, bin_d])
        final_predicted = pf.predict_from_final_scores(final_scores)
        final_score_value = final_scores[final_predicted]
        
        health_score = calculate_health_score_spectrum(multi_scores, final_scores)

        final_prediction_cache = {
            'predicted': final_predicted,
            'score': final_score_value,
            'all_scores': final_scores,
            'multi_score': multi_scores,
            'bin_h_score': bin_h,
            'bin_sh_score': bin_sh,
            'bin_d_score': bin_d,
            'health_score': health_score
            
        }
    except Exception as e:
        print(f"Warning: Could not calculate final prediction: {e}")
        final_prediction_cache = {'predicted': 'Unknown', 'score': 0.0, 'all_scores': {}}
        
    print(final_prediction_cache['all_scores'])

    


    
    # === toolbar ===
    # menubar = tk.Menu(popup)
    # popup.config(menu=menubar)
    # file_menu = tk.Menu(menubar, tearoff=0)
    # menubar.add_cascade(label="File", menu=file_menu)
    # file_menu.add_command(label="Export as PDF", 
    #                       command=export_all_models_analysis)
    # file_menu.add_separator()
    # file_menu.add_command(label="Close Window", command=popup.destroy)
    
    toolbar = tk.Frame(popup, bg="#f6f6f6", height=55)
    toolbar.pack(side="top", fill="x")

    # File Button - Larger and more prominent
    file_btn = tk.Menubutton(toolbar, text="File", 
                             font=("Segoe UI", 11, "bold"),
                             bg="#cccccc", fg="#454545",
                             activebackground="#848484", 
                             activeforeground="white",
                             relief="flat", padx=50, pady=12)
    file_btn.pack(side="left", padx=30)

    # Dropdown menu under File
    file_menu = tk.Menu(file_btn, tearoff=0, font=("Segoe UI", 10))
    file_btn["menu"] = file_menu

    file_menu.add_command(label="Export as PDF", 
                          command=export_all_models_analysis)
    file_menu.add_command(label="Save All Charts as Images", 
                          command=save_all_charts_as_images)
    file_menu.add_separator()
    file_menu.add_command(label="Close Window", command=popup.destroy)
    

    
    # 2. Setup Scrollable Region
    # Create a canvas and a scrollbar
    canvas = tk.Canvas(popup, background="#ffffff")
    scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
    
    # This frame will hold all the content
    scrollable_frame = tk.Frame(canvas, background="#ffffff")

    # Update scroll region whenever the frame size changes
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Create window inside canvas to hold the frame
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1250)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Health Score Banner at the very top
    color = "#2ecc71" if health_score >= 70 else "#f1c40f" if health_score >= 45 else "#e74c3c"
    health_frame = tk.Frame(scrollable_frame, bg=color, height=90)
    health_frame.pack(side="top", fill="x", pady=15, padx=25)

    tk.Label(health_frame, text=f"Health Status: {health_score:.1f}%", 
             font=("Helvetica", 32, "bold"), fg="white", bg=color).pack(pady=8)

    status = "Excellent Health" if health_score >= 85 else \
             "Good Health" if health_score >= 70 else \
             "Fair" if health_score >= 50 else \
             "Poor" if health_score >= 30 else "Critical"

    tk.Label(health_frame, text=status, 
             font=("Helvetica", 16, "bold"), fg="white", bg=color).pack()
   

    # 3. Loop through each model and build the UI
    analysis_cache = {}
    for model, model_name in analysis_models:
        if model is None:
            continue

        contrib_df, predicted, scores = explain_prediction(current_sample_df, model, top_n=30)
        # print(contrib_df)
        # left merge with extra data
        print("[Debug] extra_props is like this:", extra_props_df)
        if not extra_props_df.empty:
            contrib_df = contrib_df.merge(extra_props_df[['Triplet', 'pathology', 'category']],  # add any other columns you want
                left_on='triplet', 
                right_on='Triplet',
                how='left')
            # Clean up
            contrib_df = contrib_df.drop(columns=['Triplet'], errors='ignore')
            # Fill missing values
            contrib_df['pathology'] = contrib_df['pathology'].fillna("Unknown")
            contrib_df['category'] = contrib_df['category'].fillna("Unknown")
            
        analysis_cache[model_name] = (contrib_df, predicted, scores)
       
        # Section Header
        tk.Label(scrollable_frame, text=f"--- Model: {model_name} ---", 
                 font=("Helvetica", 18, "bold"), fg="#1976D2", bg="#ffffff").pack(pady=(30, 5))
        tk.Label(scrollable_frame, text=f"Predicted Class: {predicted} | Score: {scores[predicted]:.4f}", 
                 font=("Helvetica", 13, "italic", "bold"), bg="#ffffff").pack(pady=5)

        # Treeview Table
        tree = ttk.Treeview(scrollable_frame, columns=("triplet", "abundance", "contribution", "direction", "pathology", "category"), 
                            show="headings", height=8)
        tree.heading("triplet", text="Triplet")
        tree.heading("abundance", text="Abundance")
        tree.heading("contribution", text="Contribution")
        tree.heading("direction", text="Effect")
        tree.heading("pathology", text="Pathology")
        tree.heading("category", text="Category")
        
        tree.column("triplet", width=480)
        tree.column("abundance", width=90)
        tree.column("contribution", width=110)
        tree.column("direction", width=80)
        tree.column("pathology", width=140)
        tree.column("category", width=140)
        tree.pack(fill="x", padx=30, pady=10)

        for _, row in contrib_df.iterrows():
            tree.insert("", "end", values=(
                row['triplet'][:85] + "..." if len(row['triplet']) > 85 else row['triplet'],
                row['abundance'],
                row['contribution'],
                row['direction'],
                row.get('pathology', 'N/A'),
                row.get('category', 'N/A')))

        # Bar Chart
        
        
        # fig, ax = plt.subplots(figsize=(10, 5))
        fig, (ax_chart, ax_text) = plt.subplots(1,2, figsize=(14, 6), gridspec_kw={'width_ratios': [2,1]})
        colors = ['green' if c > 0 else 'red' for c in contrib_df['contribution']]
        # bars = ax.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
        # ax.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
        
        # ax.set_xlabel("Contribution (Positive = Supports Prediction)")
        # ax.set_title(f"Top {len(contrib_df)} Feature Contributions")
        # ax.invert_yaxis()
        # ax.grid(True, axis='x', alpha=0.3)
        
        # max_contrib = contrib_df['contribution'].abs().max()
        
        # # ax.set_xlim(-max_contrib * 1.1, max_contrib * 1.5)
        
        # ax.tick_params(axis='both', which='major', labelsize=8)
        
        # for bar, row in zip(bars, contrib_df.itertuples()):
        #     pathology = getattr(row, 'pathology', 'N/A')
        #     category = getattr(row, 'category', 'N/A')
        #     label = f"{pathology} | {category}"
            
        #     # Place text to the right of the bar
        #     ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
        #             label,
        #             va='center', fontsize=9, color='#333333')
            
        #     offset = max_contrib * 0.02
        #     ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
        #             label,
        #             va='center', 
        #             ha='left', 
        #             fontsize=8, 
        #             color='#333333',
        #             clip_on=False 
        #            )
        
        # plt.tight_layout()
        # plt.subplots_adjust(left=0., right=0.75, top=0.9, bottom=0.1)
        # plt.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.15)
        
        ax_chart.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
        ax_chart.set_xlabel("Contribution")
        ax_chart.set_title(f"Feature Contributions ({model_name})")
        ax_chart.invert_yaxis()
        ax_chart.grid(True, axis='x', alpha=0.3)
        ax_chart.tick_params(axis='y', labelsize=8) # Triplet names on the left

        # --- Plot 2: The Metadata (Pathology/Category) ---
        ax_text.axis('off') # Hide the box and axes for the text area
        ax_text.set_title("Pathology Details", fontsize=10, fontweight='bold')
        
        # Invert axis so the top row of text matches the top bar of the chart
        ax_text.set_ylim(ax_chart.get_ylim()) 

        for i, row in enumerate(contrib_df.itertuples()):
            pathology = getattr(row, 'pathology', 'N/A')
            # category = getattr(row, 'category', 'N/A')
            label = f"{pathology}\n" 
            
            # Place text at the same Y-coordinate as the bars
            ax_text.text(0, i, label, va='center', ha='left', 
                         fontsize=8, color='#333333')

        # Adjust layout to prevent the Triplet labels on the far left from being cut
        # plt.tight_layout()
        plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.12, wspace=0.1)
        # plt.subplots_adjust(left=0.2, right=0.3, top=0.9, bottom=0.15)

        chart_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(pady=10, padx=20)
        
        # Memory management: Close figure after drawing
        plt.close(fig)



    # 4. Final Close Button at the bottom
    tk.Button(scrollable_frame, text="Close Analysis", command=popup.destroy, 
              font=("Helvetica", 12), bg="#607D8B", fg="white", width=20).pack(pady=40)
    
    loading_frame.destroy()

    # Enable Mouse Wheel Scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-3>", show_export_context_menu)
    scrollable_frame.bind("<Button-3>", show_export_context_menu)


# ====================== PREDICTION DISPLAY ======================

def display_prediction_results(results_list):
    global all_prediction_results
    all_prediction_results = results_list
    global result_tree
    
    # Clear previous
    if result_tree is None or not result_tree.winfo_exists():
        return

    for item in result_tree.get_children():
        result_tree.delete(item)
    
    # Iterate through the list of dictionaries
    for res in results_list:
        # Access the keys defined in your pipeline function
        filename = res['filename']
        group = res['predicted_class']
        # We'll show the score of the predicted class
        score = res['final_scores'].get(group, 0.0)
        
        result_tree.insert("", "end", values=(filename, group, f"{score:.4f}"))


def on_sample_selected(event):
    global current_sample_df, current_model, current_sample_full_result
    
    selected_item = result_tree.selection()
    if not selected_item:
        return
    
    index = result_tree.index(selected_item[0])
    
    if all_prediction_results and index < len(all_prediction_results):
        res = all_prediction_results[index]
        current_sample_full_result = res # Store the whole dict
        current_sample_df = res.get('df') 
        current_model = three_group_model 
        print(f"Selected: {res['filename']} → {res['predicted_class']}")

# ====================== MAIN GUI FUNCTIONS ======================

def save_all_charts_as_images():
    """Save all 4 model charts as separate PNG images."""
    if analysis_cache is None or current_sample_df is None:
        showinfo("Error", "Please open the detailed analysis popup first!")
        return

    folder_path = fd.askdirectory(title="Select Folder to Save Charts", initialdir="Desktop/")
    if not folder_path:
        return

    try:
        filename_base = current_sample_df.attrs.get('filename', 'sample').replace('.csv', '')
        
        for model_name, (contrib_df, predicted, scores) in analysis_cache.items():
            fig, ax = plt.subplots(figsize=(12, 7))
            colors = ['green' if c > 0 else 'red' for c in contrib_df['contribution']]
            
            ax.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
            ax.set_xlabel("Contribution (Positive = Supports Prediction)")
            ax.set_title(f"{model_name}\nPredicted: {predicted} | Score: {scores[predicted]:.4f}")
            ax.invert_yaxis()
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()

            safe_name = model_name.replace(" ", "_").replace(":", "").replace("/", "")
            save_path = os.path.join(folder_path, f"{filename_base}_{safe_name}_chart.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        showinfo("Success", f"All 4 charts have been saved to:\n{folder_path}")

    except Exception as e:
        showinfo("Error", f"Failed to save charts:\n{str(e)}")
 
def start_preprocessing(self_folder_path, dest_path, full_preprocess: bool = True):
    """Start preprocessing in background thread with loading spinner"""
    global progress_bar

    # Disable buttons
    btn_choose.config(state='disabled')
    btn_analyze.config(state='disabled')

    # create grey backdrop
    overlay_frame = tk.Frame(window, bg="#B9B9B9")   # Pale grey
    overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
    # overlay_frame.attributes = None  
    # overlay_frame.lower()  
    
    overlay_label = tk.Label(overlay_frame, bg="#DFDFDF", 
                           text="", width=1000, height=1000)
    overlay_label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Create and show loading spinner
    progress_bar = ttk.Progressbar(overlay_frame, mode='indeterminate', length=300)
    progress_bar.place(relx=0.5, rely=0.5, anchor="center")   # center of window
    progress_bar.start(8)  # spinning animation

    status_label = tk.Label(overlay_frame, text="Preprocessing files... Please wait (this may take a while)", 
                            bg="#f5f385", fg="black", font=("Helvetica", 12))
    status_label.place(relx=0.5, rely=0.45, anchor="center")

    # Run heavy work in thread
    thread = threading.Thread(
        target=run_preprocessing_thread,
        args=(self_folder_path, dest_path, overlay_frame, status_label, full_preprocess),
        daemon=True
    )
    thread.start()


def run_preprocessing_thread(input_folder, dest_path, overlay_frame, status_label, full_preprocess):
    """background thread to run file preprocessing"""
    try:
        if full_preprocess:
            # 1. Convert msdos to csv
            status_label.config(text="Step 1/3: Converting files to CSV...")
            window.after(0, lambda: None)  # force update

            mtc.msdos_to_csv(input_folder, dest_path)

            # 2. Append header (overwrite)
            status_label.config(text="Step 2/3: Adding headers to CSV files...")
            ap.append_header_overwrite(dest_path)

            # 3. Crop to top N rows
            status_label.config(text="Step 3/3: Keeping only top 7434 rows...")
            ex.extract_top_n(dest_path, 7434, "Number")
            
        else: # dont need msdos to csv conversion initiate this
            # === PARTIAL PROCESS (Header + Crop only) ===
            status_label.config(text="Step 1/2: Adding headers to CSV files...")
            ap.append_header_overwrite(dest_path)

            status_label.config(text="Step 2/2: Keeping only top 7434 rows...")
            ex.extract_top_n(dest_path, 7434, "Number")

        # Success
        window.after(0, lambda: finish_preprocessing_success(dest_path, status_label, overlay_frame))

    except Exception as e:
        window.after(0, lambda: finish_preprocessing_error(str(e), status_label))


def finish_preprocessing_success(dest_path, status_label, overlay_label):
    global progress_bar
    if progress_bar:
        progress_bar.stop()
        progress_bar.destroy()
    
    status_label.destroy()
    overlay_label.destroy()
    messagebox.showinfo("Success", f"Files successfully processed and saved to:\n{dest_path}")
    
    global selected_folder
    selected_folder = dest_path
    path_label.config(text=f"Selected Folder: {dest_path}")
    
    btn_choose.config(state='normal')
    btn_analyze.config(state='normal')


def finish_preprocessing_error(error_msg, status_label):
    global progress_bar
    if progress_bar:
        progress_bar.stop()
        progress_bar.destroy()
    
    status_label.destroy()
    messagebox.showerror("Preprocessing Error", f"An error occurred:\n{error_msg}")
    
    btn_choose.config(state='normal')
    btn_analyze.config(state='normal')       
        
def select_folder():
    global selected_folder, btn_choose, btn_analyze
    folder_path = fd.askdirectory(title='Select a Folder', initialdir='Desktop/')
    if folder_path:
        
        # check file types
        all_files = os.listdir(folder_path)
        has_raw_files = any(not f.lower().endswith('.csv') for f in all_files)
        
        if has_raw_files:
            response = messagebox.askyesno("Preprocess Files?", 
                "Detected non-CSV files. Would you like to convert, add headers, and crop these to 7k rows?")
            
            if response:
                dest_path = fd.askdirectory(title='Select or Create Folder to Save Processed Files')
                
                if dest_path: 
                    # # call function to convert msdos files to suitable state
                    # # needs to add function here
                    
                    # #1. turn files into csv
                    # mtc.msdos_to_csv(folder_path, dest_path)
                    
                    # #2. check if header exists
                    # ap.append_header_overwrite(dest_path)
                    
                    # #3. crop to only 7k entries
                    # ex.extract_top_n(dest_path, 7434, "Number")
                    
                    # selected_folder = dest_path
                    # messagebox.showinfo("Success", f"Processed files saved to:\n{dest_path}")
                    # path_label.config(text=f"Selected Folder: {dest_path}")
                    start_preprocessing(folder_path, dest_path, full_preprocess=True)
                else:
                    messagebox.showwarning("Cancelled", "No destination selected.")
            
            else:
                selected_folder = folder_path
                messagebox.showwarning("Caution, ""File type may not be compatible with program requirement and cause error if chose to proceed. If persist, false results may be returned or error code would appear.")
            
        else: # when all files are csv
            # check if they all have headers and only top7k 
            
            # Expected header
            expected_header = [
                "Number", "GeneV", "GeneJ", "Protein", "Percent", "Total",
                "annotation", "Epitope_species_DB_V1.0 sapiens/HomoSapiens/Wheat sapiens/Wheat"
            ]
            
            num_headers = len(expected_header)

            folder_path = Path(folder_path)
    
            csv_files = list(folder_path.glob("*.csv"))
            
            print(f"Found {len(csv_files)} CSV files. Processing...\n")
            
            # add lock to prevent user from processing during check
            btn_choose.config(state='disabled')
            btn_analyze.config(state='disabled')
            # path_label.config(text="Checking files... please wait.")
            window.update_idletasks() # Force the GUI to refresh visually 
            for csv_file in csv_files:
                try:
                    # Step 1: Read first few rows to detect header
                    df_sample = pd.read_csv(csv_file, nrows=5, header=None, dtype=str)
                    
                    first_row = df_sample.iloc[0].astype(str).str.strip().tolist()
                    
                    # Clean expected header for comparison
                    expected_clean = [h.strip() for h in expected_header]
                    
                    # Check if first row looks like our expected header (case-insensitive, allow minor differences)
                    has_header = False
                    if len(first_row) >= num_headers - 2:  # allow small column difference
                        # Compare first few important columns
                        match_count = sum(
                            any(word in cell or cell in word for word in ["Number", "GeneV", "GeneJ", "Protein", "Percent"])
                            for cell in first_row[:5]
                        )
                        if match_count >= 3:  # quite strict
                            has_header = True

                    # Now read the full file properly
                    if has_header:
                        # df = pd.read_csv(csv_file, header=0, dtype=str)
                        print(f"  → Detected header in {csv_file.name}")
                    else:
                        # ask user path to store processed csv files
                        response = messagebox.askyesno("Preprocess Files?", 
                            "Detected no header csv files. Would you like to add headers and crop these to 7k rows?")
                        
                        if response:
                            # ask for path
                            dest_path = fd.askdirectory(title='Select or Create Folder to Save Processed Files')
                            
                            if dest_path:
                                # call function to append header and crop to 7k
                                
                                # 1. append header
                                # ap.append_header_overwrite(dest_path)
                                
                                # # 2. crop to 7k
                                # ex.extract_top_n(dest_path, 7434, "Number")
                                
                                
                                # selected_folder = dest_path
                                # messagebox.showinfo("Success", f"Processed files saved to:\n{dest_path}")
                                # path_label.config(text=f"Selected Folder: {dest_path}")
                                start_preprocessing(folder_path, dest_path, full_preprocess=False)
                            
                            else:
                                messagebox.showwarning("Cancelled", "No destination selected.")
            
                        else:
                            selected_folder = folder_path
                            messagebox.showwarning("File type may not be compatible with program requirement and cause error if chose to proceed.")        
                        break
                except Exception as e:
                    print(f"✗ Error processing {csv_file.name}: {e}")
                
                finally:
                    # 3. UNLOCK UI (Always runs)
                    if btn_choose and btn_analyze:
                        btn_choose.config(state='normal')
                        btn_analyze.config(state='normal')
            
        selected_folder = folder_path
        path_label.config(text=f"Selected Folder: {folder_path}")
        
        
def display_prediction_summary(results: list):
    """Display summary of predictions"""
    from collections import Counter
    count = Counter(r['predicted_class'] for r in results)
    total = len(results)
    
    summary = f"""
=== PREDICTION SUMMARY ===
Total Samples : {total}
Healthy       : {count.get('healthy', 0)} ({count.get('healthy', 0)/total*100:.1f}%)
Subhealthy    : {count.get('subhealthy', 0)} ({count.get('subhealthy', 0)/total*100:.1f}%)
Disease       : {count.get('disease', 0)} ({count.get('disease', 0)/total*100:.1f}%)
"""
    print(summary)
    # messagebox.showinfo("Analysis Summary", summary)

def analyze_files():
    count_disease = 0
    count_shealthy = 0
    count_healthy = 0
    if not selected_folder:
        showinfo("No folder", "Please select a folder first.")
        return
    
    # clear_window()

    analyze_label = tk.Label(window, text="Analyzing...", bg="#f5f385", fg="black", font=("Helvetica", 14))
    analyze_label.grid(row=0, column=0, sticky="nsew")
    window.update()

    try:
        df_check = pf.build_dataframe_verification(selected_folder)
        if not df_check:
            analyze_label.config(text="No valid files found.")
            window.after(2000, analyze_label.destroy)
            return

        print("DEBUG: Starting prediction...")

        global results, top_n, threshold_mode_used
        results = pf.predict_df_class(
            df_check, 
            three_group_model, 
            binary_shealthy_model, 
            binary_healthy_model, 
            binary_disease_model,
            top_n,
            threshold_mode_used
            
        )
        
        display_prediction_summary(results)

        print(f"DEBUG: predict_df_class returned type = {type(results)}")

        if results is None or not isinstance(results, list):
            raise ValueError(f"predict_df_class returned {type(results)} instead of list!")

        display_prediction_results(results)
        analyze_label.destroy()

    except Exception as e:
        analyze_label.destroy()
        showinfo("Error", f"Analysis failed:\n\n{str(e)}")
        


def show_export_context_menu(event):
    """Reusable right-click menu for both the main table and the popup."""
    menu = tk.Menu(window, tearoff=0)
    menu.add_command(label="Export Detailed Analysis", 
                     command=export_all_models_analysis)
    menu.post(event.x_root, event.y_root)
    
    
def center_popup(popup, parent):
    popup.update_idletasks()  # Ensures geometry is calculated correctly
    
    # Get dimensions of the popup and the parent window
    w = 500
    h = 180
    
    # Get parent window position and size
    parent_x = parent.winfo_rootx()
    parent_y = parent.winfo_rooty()
    parent_w = parent.winfo_width()
    parent_h = parent.winfo_height()

    # Calculate the center point
    x = parent_x + (parent_w // 2) - (w // 2)
    y = parent_y + (parent_h // 2) - (h // 2)

    popup.geometry(f"{w}x{h}+{x}+{y}")
    
def export_all_models_analysis():
    """Export what is shown in the detailed explanation popup into one PDF."""
    global analysis_cache, final_prediction_cache, current_sample_df
    if analysis_cache is None:
        showinfo("Error", "Please open a sample from the results table first.")
        return

    # The same 4 models used in your show_explanation()
    # analysis_models = [
    #     (three_group_model, "Ensemble (3-Group)"),
    #     (binary_healthy_model, "Binary: Healthy"),
    #     (binary_shealthy_model, "Binary: Sub-Healthy"),
    #     (binary_disease_model, "Binary: Disease")
    # ]
    
    if not current_sample_df.empty:
        filename_base = current_sample_df.attrs.get('filename', 'sample').replace('.csv', '')
    
    else:
        showinfo("Error", "No sample selected. Please select a sample first.")
        return
    
    health_score = final_prediction_cache.get('health_score', None)
    if health_score is None:
        # Fallback (should not happen)
        health_score = calculate_health_score_spectrum(
            final_prediction_cache.get('multi_scores', {}), 
            final_prediction_cache.get('all_scores', {})
        )

    # Ask user where to save
    file_path = fd.asksaveasfilename(
        title="Save Full 4-Model Analysis as PDF",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialdir="Desktop/",
        initialfile= f"{filename_base}.pdf"
    )
    
    if not file_path:
        return
    
    loading_win = tk.Toplevel(window)
    loading_win.title("Exporting PDF")
    loading_win.geometry("500x180")
    loading_win.resizable(False, False)
    center_popup(loading_win, window)
    
    loading_win.transient(window)  # Stay on top
    loading_win.grab_set()         

    tk.Label(loading_win, text="Exporting PDF Report...\nThis may take a few seconds.", 
             font=("Helvetica", 12), pady=15).pack()

    progress = ttk.Progressbar(loading_win, mode='indeterminate', length=380)
    progress.pack(pady=10)
    progress.start(15)

    status_label = tk.Label(loading_win, text="Preparing document...", font=("Helvetica", 10))
    status_label.pack(pady=8)

    loading_win.update()

    try:
        with PdfPages(file_path) as pdf:
            # ==================== Exporting Bar Charts ====================
            final_pred = current_sample_full_result['predicted_class']
            final_score = current_sample_full_result['final_scores'].get(final_pred, 0.0)
            title_count = 0
            
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.96, "TCR Repertoire Detailed Analysis Report", 
                    fontsize=20, fontweight='bold', ha='center')
            ax.text(0.5, 0.91, f"Patient File: {filename_base}", 
                    fontsize=13, ha='center')

            # Colored Banner (exactly like popup)
            color = "#2ecc71" if health_score >= 75 else "#f1c40f" if health_score >= 45 else "#e74c3c"
            ax.add_patch(plt.Rectangle((0.08, 0.58), 0.84, 0.24, color=color, alpha=0.95))

            # Big Health Percentage
            ax.text(0.5, 0.735, f"{health_score:.1f}%", 
                    fontsize=45, fontweight='bold', color='white', ha='center')

            # "Health Status" label
            ax.text(0.5, 0.665, "Health Status", 
                    fontsize=15, color='white', ha='center')

            # Status Text (Excellent / Good / Fair etc.)
            status_text = "Excellent Health" if health_score >= 85 else \
                          "Good Health" if health_score >= 70 else \
                          "Fair" if health_score >= 50 else \
                          "Poor" if health_score >= 30 else "Critical"
            
            ax.text(0.5, 0.625, status_text, 
                    fontsize=17, fontweight='bold', color='white', ha='center')
            # plt.suptitle(f"4-Model Detailed Analysis Report\n\n"
            #              f"Patient File:  {filename_base}\n "
            #              f"Final Predicted Group :  {final_pred.upper()}\n")
            # color = "#2ecc71" if health_score >= 70 else "#f1c40f" if health_score >= 45 else "#e74c3c"
            # ax.add_patch(plt.Rectangle((0.1, 0.68), 0.8, 0.15, color=color, alpha=0.9))
            # status = "Excellent Health" if health_score >= 85 else \
            # "Good Health" if health_score >= 70 else \
            # "Fair" if health_score >= 50 else \
            # "Poor" if health_score >= 30 else "Critical"
            
            # ax.text(0.5, 0.73, f"{health_score:.1f}%", 
            #         fontsize=25, fontweight='bold', color='white', ha='center')
            # ax.text(0.5, 0.61, status, 
            #         fontsize=16, fontweight='bold', color='white', ha='center')
            # circle = plt.Circle((0.5, 0.62), 0.22, color='#e0e0e0', fill=True)
            # ax.add_patch(circle)
            
            # progress_angle = 360 * (health_score / 100)
            # wedge = plt.matplotlib.patches.Wedge(
            #     center=(0.5, 0.62), r=0.22, theta1=90, theta2=90 - progress_angle,
            #     color=color, width=0.08
            # )
            # ax.add_patch(wedge)

            # # Inner white circle
            # inner_circle = plt.Circle((0.5, 0.62), 0.14, color='white', fill=True)
            # ax.add_patch(inner_circle)

            # ax.text(0.5, 0.77, f"Health Status: {health_score:.1f}%", 
            #         fontsize=26, fontweight='bold', color='white', ha='center')
            


            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
           
            # Health Score Banner (Big and prominent)
            for i, (model_name, (contrib_df, predicted, scores)) in enumerate(analysis_cache.items()): # iterate to export out all 4 model graphs
                # if model is None:
                #     continue
                    
                # contrib_df, predicted, scores = explain_prediction(
                #     current_sample_df, model, top_n=30   
                # )
              
                
                # fig, ax = plt.subplots(figsize=(15, 15))
                
                # # ax = axes[i]
                # colors = ['green' if c > 0 else 'red' for c in contrib_df['contribution']]
                
                # ax.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
                # ax.set_title(f"{model_name}\nPredicted: {predicted} (Score: {scores[predicted]:.4f})", 
                #              fontsize=13, pad = 5)
                # ax.set_xlabel("Contribution")
                # ax.invert_yaxis()
                # ax.grid(True, axis='x', alpha=0.3)
                # if title_count == 0:
                #     plt.suptitle(f"4-Model Detailed Analysis Report\n\n"
                #          f"Patient File:  {filename_base}\n "
                #          f"Final Predicted Group :  {final_pred.upper()}\n"
                #          f"Final Ensemble Score  :  {final_score:.4f}",
                #          fontsize=20, y=0.98, ha='center', fontweight = 'bold')
                #     title_count  += 1
                    
                #     # fig, ax = plt.subplots(figsize=(11, 8.5))
                #     # ax.axis('off')
                #     # color = "#2ecc71" if health_score >= 70 else "#f1c40f" if health_score >= 45 else "#e74c3c"
                #     # ax.add_patch(plt.Rectangle((0.1, 0.68), 0.8, 0.15, color=color, alpha=0.9))

                #     # ax.text(0.5, 0.77, f"Health Status: {health_score:.1f}%", 
                #     #         fontsize=26, fontweight='bold', color='white', ha='center')
                    
                #     # status = "Excellent Health" if health_score >= 85 else \
                #     #         "Good Health" if health_score >= 70 else \
                #     #         "Fair" if health_score >= 50 else \
                #     #         "Poor" if health_score >= 30 else "Critical"
                    
                #     # ax.text(0.5, 0.71, status, 
                #     #         fontsize=16, fontweight='bold', color='white', ha='center')

                #     # # pdf.savefig(fig, bbox_inches='tight')
                #     # # plt.close(fig)
                # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                # pdf.savefig(fig, dpi=300)
                # plt.close(fig)
                
                fig, (ax_chart, ax_text) = plt.subplots(1, 2, figsize=(11, 8.5), 
                                                        gridspec_kw={'width_ratios': [2.5, 1]})
                
                # --- Plot 1: The Bar Chart (Left Side) ---
                colors = ['green' if c > 0 else 'red' for c in contrib_df['contribution']]
                ax_chart.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
                
                ax_chart.set_title(f"{model_name}\nPredicted: {predicted} (Score: {scores[predicted]:.4f})", 
                                   fontsize=12, pad=10)
                ax_chart.set_xlabel("Contribution", fontsize=10)
                ax_chart.invert_yaxis()
                ax_chart.grid(True, axis='x', alpha=0.3)
                ax_chart.tick_params(axis='y', labelsize=7) # Small text for long triplets
                ax_chart.tick_params(axis='x', labelsize=8)

                # --- Plot 2: The Pathology Metadata (Right Side) ---
                ax_text.axis('off')
                ax_text.set_title("Pathology Details", fontsize=10, fontweight='bold', pad=10)
                
                # Align the coordinate systems
                ax_text.set_ylim(ax_chart.get_ylim()) 

                for idx, row in enumerate(contrib_df.itertuples()):
                    pathology = getattr(row, 'pathology', 'N/A')
                    category = getattr(row, 'category', 'N/A')
                    
                    # Formatting text for PDF (one row per bar)
                    label = f"{pathology}\n({category})"
                    
                    ax_text.text(0, idx, label, 
                                 va='center', ha='left', 
                                 fontsize=6.5, # Slightly smaller for PDF density
                                 color='#333333')

                # Main header for the page (only on first chart page or every page)
                if i == 0:
                     plt.suptitle(f"4-Model Detailed Analysis: {filename_base}\n"
                                  f"Final Result: {final_pred.upper()} ({health_score:.1f}%)", 
                                  fontsize=16, fontweight='bold', y=0.98)

                # Adjust layout so elements don't overlap or fall off page
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Save this page
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

            # plt.suptitle(f"4-Model Feature Analysis\n\n"
            #              f"Patient: {filename_base}\n", fontsize=16, y=0.98)
           
           

            # ==================== FULL TABLES (one model per section, unlimited rows) ====================
            for model_name, (contrib_df, predicted, scores) in analysis_cache.items():
                full_df = contrib_df.copy()
                # if model is None:
                #     continue
                    
                # contrib_df, predicted, scores = explain_prediction(
                #     current_sample_df, model, top_n=20   # chart uses top 20
                # )
                
                # full_df = contrib_df.copy()   # we will use the full data for the table

                # Split table into readable pages (40 rows per page)
                rows_per_page = 35
                for start in range(0, len(full_df), rows_per_page):
                    chunk = full_df.iloc[start:start + rows_per_page]
                    
                    fig, ax = plt.subplots(figsize=(14, 11))
                    ax.axis('off')
                    
                    table_data = chunk[['triplet', 'abundance', 'contribution', 'direction','pathology', 'category']].values
                    
                    table = ax.table(
                        cellText=table_data,
                        colLabels=['Triplet', 'Abundance', 'Contribution', 'Direction', 'Pathology', 'Category'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.48, 0.09, 0.09, 0.09, 0.15, 0.15]
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.0, 1.7) # vertical spacing for rows
                    
                    # ax.set_title(f"{model_name} - Full Contribution Table "
                    #              f"(rows {start+1}–{start+len(chunk)})\n"
                    #              f"Predicted: {predicted} | Score: {scores[predicted]:.4f}\n", 
                    #              fontsize=13, pad=20)
                    plt.suptitle(f"{model_name} - Full Contribution Table "
                                 f"(rows {start+1}–{start+len(chunk)})\n"
                                 f"Predicted: {predicted} | Score: {scores[predicted]:.4f}\n", 
                                 fontsize=13, y=0.96)
                    plt.subplots_adjust(top = 0.90)
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.close(fig)

 

    except Exception as e:
        showinfo("Export Failed", f"Error during PDF export:\n{str(e)}")
    
    finally:
        # Close loading window
        loading_win.destroy()
        showinfo("Export Successful", 
        f"Full 4-model analysis exported as PDF:\n\n{file_path}\n\n"
        f"• All 4 bar charts (exactly as in popup)\n"
        f"• Complete contribution tables for every model")
        
        
def export_detailed_analysis():
    """Export the exact same chart + full table from the explanation popup as a clean PDF."""
    if current_sample_df is None or current_model is None:
        showinfo("Error", "Please select a sample first!")
        return

    # Ask user where to save
    file_path = fd.asksaveasfilename(
        title="Save Detailed Analysis as PDF",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialdir="Desktop/",
        initialfile="detailed_analysis.pdf"
    )
    
    if not file_path:
        return

    try:
        contrib_df, predicted, scores = explain_prediction(
            current_sample_df, current_model, top_n=20   
        )

        filename_base = current_sample_df.attrs.get('filename', 'sample').replace('.csv', '')

        with PdfPages(file_path) as pdf:
            # ==================== PAGE 1: CHART (exactly like popup) ====================
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            
            colors = ['green' if c > 0 else 'red' for c in contrib_df['contribution']]
            
            ax1.barh(contrib_df['triplet'], contrib_df['contribution'], color=colors)
            ax1.set_xlabel("Contribution to Final Score (positive = supports prediction)")
            ax1.set_title(f"Feature Contributions - Predicted: {predicted.upper()}\n"
                          f"Final Score: {scores[predicted]:.4f}\n"
                          f"Patient: {filename_base}", fontsize=14, pad=20)
            ax1.invert_yaxis()
            ax1.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig1, dpi=300, bbox_inches='tight')
            plt.close(fig1)

            # ==================== PAGE 2+: FULL TABLE (no limit) ====================
            full_table_df = contrib_df.copy()  # Full data, not just top 30

            # Split table into chunks of 40 rows per page (readable size)
            rows_per_page = 40
            for start in range(0, len(full_table_df), rows_per_page):
                chunk = full_table_df.iloc[start:start + rows_per_page]
                
                fig2, ax2 = plt.subplots(figsize=(14, 10))
                ax2.axis('off')
                
                table_data = chunk[['triplet', 'abundance', 'contribution', 'direction']].values
                
                table = ax2.table(
                    cellText=table_data,
                    colLabels=['Triplet', 'Abundance', 'Contribution', 'Direction'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.55, 0.15, 0.15, 0.15]
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.8)
                
                ax2.set_title(f"Full Contribution Table (rows {start+1}-{start+len(chunk)})", 
                              fontsize=14, pad=20)
                
                pdf.savefig(fig2, dpi=300, bbox_inches='tight')
                plt.close(fig2)

        # showinfo("Export Successful", 
        #         f"Detailed analysis saved as PDF:\n\n{file_path}\n\n"
        #         f"Contains:\n"
        #         f"• Contribution Chart (exact same as popup)\n"
        #         f"• Full Contribution Table (all rows)")

    except Exception as e:
        showinfo("Export Failed", f"Error during PDF export:\n{str(e)}")

def clear_window():
    for widget in window.winfo_children():
        widget.destroy()


def show_context_menu(event):
    """Show context menu when user right-clicks on a row in the Treeview."""
    # Get the row that was clicked
    item = result_tree.identify_row(event.y)
    if not item:
        return
    
    # Select the row
    result_tree.selection_set(item)
    
    # Trigger the selection handler so current_sample_df is updated
    on_sample_selected(None)
    
    # Create context menu
    menu = tk.Menu(window, tearoff=0)
    menu.add_command(label="Export Detailed Analysis", 
                     command=export_all_models_analysis)
    
    # Show the menu at mouse position
    menu.post(event.x_root, event.y_root)

def create_main_menu():
    global path_label, result_tree, btn_choose, btn_analyze
    
    content_wrapper = tk.Frame(window, bg="#ffffff")
    content_wrapper.grid(row=0, column=0, sticky="nsew", padx=50, pady=20)
    
    path_label = tk.Label(content_wrapper, text="Selected Folder: None", font=("Helvetica", 14))
    path_label.pack(pady=30)

    # Buttons
    button_frame = tk.Frame(content_wrapper, bg="#ffffff")
    button_frame.pack(pady=40)

    btn_choose = tk.Button(button_frame, text="Choose Folder", font=("Helvetica", 15), bg="#959595", fg="#202020",
              width=18, height=2, command=select_folder)
    btn_choose.grid(row=0, column=0, padx=20)

    btn_analyze = tk.Button(button_frame, text="Analyze", font=("Helvetica", 15), bg="#6BED5F", fg="#202020",
              width=18, height=2, command=analyze_files)
    btn_analyze.grid(row=0, column=1, padx=20)
    
    

    # Treeview for results
    global result_tree
    
    style = ttk.Style()

    # Set font family and size for the data rows
    style.configure("Treeview", font=("Arial", 11), rowheight=30)
    
    result_tree = ttk.Treeview(content_wrapper, columns=("filename", "prediction", "score"), show="headings", height=10)
    result_tree.heading("filename", text="Filename")
    result_tree.heading("prediction", text="Predicted Group")
    result_tree.heading("score", text="Final Score")
    
    result_tree.column("filename", width=400, stretch=True)
    result_tree.column("prediction", anchor="center", width = 200, stretch = True)
    result_tree.column("score", anchor="center", width = 150, stretch =True)
    
    result_tree.pack(pady=20, padx=40, fill="both", expand=True)
    
    result_tree.bind("<<TreeviewSelect>>", on_sample_selected)
    
    result_tree.bind("<Double-1>", lambda event: show_explanation())
    result_tree.bind("<Button-3>", show_export_context_menu)   

    # Explain button



# ====================== INITIALIZATION ======================
# ====================== THREADED INITIALIZATION ======================

def start_initialization():
    """Start model loading with overlay + spinner"""
    global progress_bar, overlay_frame, initialization_label

    # Create grey overlay
    overlay_frame = tk.Frame(window, bg="#D0D0D0")
    overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    # Status label
    initialization_label = tk.Label(overlay_frame, 
                                    text="Initializing Models... This may take 10-30 seconds",
                                    bg="#f5f385", fg="black", 
                                    font=("Helvetica", 13, "bold"), padx=30, pady=15)
    initialization_label.place(relx=0.5, rely=0.42, anchor="center")

    # Progress spinner
    progress_bar = ttk.Progressbar(overlay_frame, mode='indeterminate', length=450)
    progress_bar.place(relx=0.5, rely=0.48, anchor="center")
    progress_bar.start(8)

    # Run heavy initialization in background thread
    thread = threading.Thread(target=run_initialization_thread, daemon=True)
    thread.start()

def safe_update_status(new_text: str):
    """Safely update status label from any thread + force redraw"""
    def _update():
        if initialization_label and initialization_label.winfo_exists():
            initialization_label.config(text=new_text)
            window.update_idletasks()          # ← This forces immediate refresh
    window.after(0, _update)

def run_initialization_thread():
    """All heavy model loading runs here"""
    global three_group_model, binary_disease_model, binary_shealthy_model, binary_healthy_model, extra_props_df

    try:
        safe_update_status("Loading feature sets from CSV...")
        # Update status from thread safely
        window.after(0, lambda: initialization_label.config(
            text="Loading feature sets..."))

        # three-group model
        features_set_full_external = pf.load_selected_features(
            "models/7k_three_group_new_mi_left_merge.csv"
        )
        features_full_external_filtered = features_set_full_external[
            features_set_full_external['combined_score'] > 0.02
        ].sort_values(by='combined_score', ascending=False)
        

        window.after(0, lambda: initialization_label.config(text="Building 3-Group Model..."))
        three_group_model = pf.build_tscore_model_from_df(features_full_external_filtered)
        #[extra_info]
        #print(features_full_external_filtered.head(3))
        #export_filtered_features(features_full_external_filtered, "3group_filtered_features.csv")
        
        # Binary Disease Model
        window.after(0, lambda: initialization_label.config(text="Building Binary Disease Model..."))
        feature_binary_disease = pf.load_selected_features_binary_disease(
            "models/7k_binary_disease_training_new_mi_left_merge.csv"
        )
        features_binary_disease_filtered = feature_binary_disease[
            (feature_binary_disease['combined_score'] > 0.06) & 
            (feature_binary_disease['Hydrophobicity'] > -0.9) &
            (feature_binary_disease['Helix/extension propensity'] > -0.2) & 
            (feature_binary_disease['d_disease'] > 0.20)
        ].sort_values(by='combined_score', ascending=False).head(2500)
        binary_disease_model = pf.build_tscore_model_binary_from_df_disease(features_binary_disease_filtered)
        #[extra_info]
        #export_filtered_features(features_binary_disease_filtered, "binary_disease_filtered_features.csv")

        # Binary Healthy Model
        window.after(0, lambda: initialization_label.config(text="Building Binary Healthy Model..."))
        feature_binary_healthy = pf.load_selected_features_binary_healthy(
            "models/7k_binary_healthy_training_new_mi_left_merge.csv"
        )
        features_binary_healthy_filtered = feature_binary_healthy[
            (feature_binary_healthy['combined_score'] > 0.05) & 
            (feature_binary_healthy['Hydrophobicity'] < 0.05) & 
            (feature_binary_healthy['Helix/extension propensity'] > -0.15) &
            (feature_binary_healthy['d_healthy'] > 0.18) &
            (feature_binary_healthy['pathology'] == 'none')
        ].sort_values(by='combined_score', ascending=False).head(1500)
        binary_healthy_model = pf.build_tscore_model_binary_from_df_healthy(features_binary_healthy_filtered)
         #[extra_info]
        export_filtered_features(features_binary_healthy_filtered, "binary_healthy_filtered_features.csv")

        # Binary Subhealthy Model
        window.after(0, lambda: initialization_label.config(text="Building Binary Sub-Healthy Model..."))
        feature_binary_shealthy = pf.load_selected_features_binary_shealthy(
            "models/7k_binary_subhealthy_training_new_mi_left_merge.csv"
        )
        features_binary_shealthy_filtered = feature_binary_shealthy[
            (feature_binary_shealthy['combined_score'] > 0.06) &  
            (feature_binary_shealthy['d_other'] < 0.25) &
            (feature_binary_shealthy['d_subhealthy'] > 0.196)
        ].sort_values(by='combined_score', ascending=False)
        binary_shealthy_model = pf.build_tscore_model_binary_from_df_shealthy(features_binary_shealthy_filtered)
        #[extra_info]
        #export_filtered_features(features_binary_shealthy_filtered, "binary_subhealthy_filtered_features.csv")
        
        # jia jie's data
        relative_path = Path("models/V1_TCR_Dataset_preprocessed.csv")
        extra_props_df = pd.read_csv(relative_path)
        print(f"[INFO] Extra properties loaded: {len(extra_props_df)} rows")

        # Success
        window.after(0, finish_initialization_success)

    except Exception as e:
        print(f"Initialization error: {e}")
        window.after(0, lambda err=e: finish_initialization_error(str(err)))

# [code to get extra information for analyze, not used in launch version]
def export_filtered_features(df: pd.DataFrame, filename: str):
    """Export the filtered features."""
    if df is None or df.empty:
        print(f"[WARN] No features to export for {filename}")
        return
    
    # Make sure 'triplet' is a column (not index)
    export_df = df.copy()
    
    if 'Triplet' not in export_df.columns:
        if export_df.index.name == 'Triplet' or 'Triplet' in export_df.index.names:
            export_df = export_df.reset_index()
        else:
            # Try to find triplet column with different possible names
            possible_names = ['triplet', 'Triplet', 'feature', 'CDR3', 'Protein']
            for name in possible_names:
                if name in export_df.columns:
                    export_df = export_df.rename(columns={name: 'triplet'})
                    break
    
    # Ensure triplet is the first column
    if 'Triplet' in export_df.columns:
        cols = ['Triplet'] + [col for col in export_df.columns if col != 'Triplet']
        export_df = export_df[cols]

    # Create export directory
    export_dir = Path(r"C:\Users\ASUS\OneDrive\桌面\FYP_resources\found_triplets")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = export_dir / filename
    export_df.to_csv(output_path, index=False)
    
    print(f"[INFO] Exported {len(export_df)} features with triplet column → {output_path}")
    
def finish_initialization_success():
    cleanup_overlay()
    messagebox.showinfo("Success", "All models loaded successfully!")
    create_main_menu()


def finish_initialization_error(error_msg):
    cleanup_overlay()
    messagebox.showerror("Initialization Failed", 
                        f"Failed to load models:\n\n{error_msg}\n\nPlease check your model files.")
    
def cleanup_overlay():
    global progress_bar, overlay_frame, initialization_label
    if progress_bar and progress_bar.winfo_exists():
        progress_bar.stop()
        progress_bar.destroy()
        progress_bar = None
    if overlay_frame and overlay_frame.winfo_exists():
        overlay_frame.destroy()
        overlay_frame = None
    initialization_label = None

def initialization():
    start_initialization()

# ====================== MAIN ======================

window = tk.Tk()
window.title("Prediction System")
window.geometry("1280x720")
window.configure(background="#ffffff")
window.resizable(True, True)
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

main_app_frame = tk.Frame(window, bg="#ffffff")
main_app_frame.grid(row=0, column=0, sticky="nsew")
main_app_frame.columnconfigure(0, weight=1)
initialization()
window.mainloop()