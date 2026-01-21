"""
Batch Behavior Analysis Tool
Analyzes multiple CSV files from different groups and individuals
Generates comparison visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime


class BatchAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Behavior Analyzer")
        self.root.geometry("1000x700")
        sns.set_theme(style="whitegrid")
        
        # Data storage
        self.csv_files = []  # List of (filepath, group, animal_id)
        self.data_list = []  # List of DataFrames
        
        self.build_gui()
    
    def build_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="CSV Files", padding="10")
        file_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=10)
        self.file_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(btn_frame, text="Add CSV Files", command=self.add_files).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_files).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected).pack(side='left', padx=2)
        
        # Analysis buttons
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding="10")
        analysis_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(analysis_frame, text="Show Timeline Comparison", 
                  command=self.show_timeline).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Show Duration Distribution", 
                  command=self.show_duration_dist).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Show Group Statistics", 
                  command=self.show_group_stats).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Show Group Total Time", 
                  command=self.show_group_total).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Export Summary Report", 
                  command=self.export_report).pack(fill='x', pady=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Add CSV files to analyze")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                relief='sunken', anchor='w')
        status_label.pack(fill='x')
    
    def add_files(self):
        """Add CSV files with group assignment"""
        filenames = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filenames:
            return
        
        for filepath in filenames:
            # Extract animal_id from filename (e.g., "M01_grooming.csv" -> "M01")
            basename = os.path.basename(filepath)
            if '_' in basename:
                animal_id = basename.split('_')[0]
            else:
                animal_id = basename.replace('.csv', '')
            
            # Ask for group assignment
            group = self.ask_group(animal_id)
            if group is None:
                continue
            
            self.csv_files.append((filepath, group, animal_id))
            display_name = f"{group} - {animal_id} - {basename}"
            self.file_listbox.insert('end', display_name)
        
        self.status_var.set(f"Loaded {len(self.csv_files)} files")
    
    def ask_group(self, animal_id):
        """Dialog to assign group"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Assign Group for {animal_id}")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = [None]
        
        ttk.Label(dialog, text=f"Animal ID: {animal_id}", font=('Arial', 10, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="Enter Group Name:").pack()
        
        group_var = tk.StringVar(value="Control")
        entry = ttk.Entry(dialog, textvariable=group_var, width=20)
        entry.pack(pady=5)
        entry.focus()
        
        def confirm():
            result[0] = group_var.get().strip()
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=confirm).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
        
        entry.bind('<Return>', lambda e: confirm())
        
        dialog.wait_window()
        return result[0]
    
    def clear_files(self):
        self.csv_files = []
        self.data_list = []
        self.file_listbox.delete(0, 'end')
        self.status_var.set("All files cleared")
    
    def remove_selected(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        for index in reversed(selection):
            self.csv_files.pop(index)
            self.file_listbox.delete(index)
        
        self.status_var.set(f"{len(self.csv_files)} files remaining")
    
    def load_data(self):
        """Load all CSV files"""
        if not self.csv_files:
            messagebox.showwarning("No Files", "Please add CSV files first.")
            return None
        
        self.data_list = []
        for filepath, group, animal_id in self.csv_files:
            try:
                df = pd.read_csv(filepath)
                df['group'] = group
                df['file_animal_id'] = animal_id
                self.data_list.append(df)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {filepath}:\n{str(e)}")
                return None
        
        if not self.data_list:
            return None
        
        return pd.concat(self.data_list, ignore_index=True)
    
    def show_timeline(self):
        """Show timeline comparison across all individuals"""
        df_all = self.load_data()
        if df_all is None:
            return
        
        window = tk.Toplevel(self.root)
        window.title("Timeline Comparison")
        window.geometry("1200x800")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Sort by group and animal
        df_all = df_all.sort_values(['group', 'file_animal_id'])
        
        # Assign y-position for each individual
        individuals = df_all.groupby(['group', 'file_animal_id']).size().index.tolist()
        y_positions = {ind: i for i, ind in enumerate(individuals)}
        
        # Color by group
        groups = df_all['group'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        group_colors = {g: colors[i] for i, g in enumerate(groups)}
        
        # Plot each event
        for _, row in df_all.iterrows():
            y = y_positions[(row['group'], row['file_animal_id'])]
            color = group_colors[row['group']]
            ax.barh(y, row['duration_s'], left=row['start_s'], height=0.8,
                   color=color, edgecolor='black', alpha=0.7)
        
        # Labels
        ax.set_yticks(range(len(individuals)))
        ax.set_yticklabels([f"{grp} - {aid}" for grp, aid in individuals])
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Individual', fontsize=12)
        ax.set_title('Behavior Timeline Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=group_colors[g], label=g) for g in groups]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Save image button
        toolbar = ttk.Frame(window)
        toolbar.pack(fill='x', pady=5)
        def save_fig():
            fname = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")]
            )
            if fname:
                try:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image saved to:\n{fname}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        ttk.Button(toolbar, text="Save Image...", command=save_fig).pack(side='right')
        
        self.status_var.set("Timeline displayed")
    
    def show_duration_dist(self):
        """Show duration distribution by group"""
        df_all = self.load_data()
        if df_all is None:
            return
        
        window = tk.Toplevel(self.root)
        window.title("Duration Distribution")
        window.geometry("1000x600")
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        groups = df_all['group'].unique()
        for group in groups:
            data = df_all[df_all['group'] == group]['duration_s']
            ax.hist(data, bins=15, alpha=0.6, label=group, edgecolor='black')
        
        ax.set_xlabel('Duration (s)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Behavior Duration Distribution by Group', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Save image button
        toolbar = ttk.Frame(window)
        toolbar.pack(fill='x', pady=5)
        def save_fig():
            fname = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")]
            )
            if fname:
                try:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image saved to:\n{fname}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        ttk.Button(toolbar, text="Save Image...", command=save_fig).pack(side='right')
        
        self.status_var.set("Duration distribution displayed")
    
    def show_group_stats(self):
        """Show statistical comparison between groups"""
        df_all = self.load_data()
        if df_all is None:
            return
        
        window = tk.Toplevel(self.root)
        window.title("Group Statistics")
        window.geometry("900x700")
        
        # Calculate statistics
        stats = df_all.groupby('group')['duration_s'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('total', 'sum')
        ]).round(2)
        
        # Create figure with subplots
        fig = Figure(figsize=(9, 7), dpi=100)
        
        # Box plot
        ax1 = fig.add_subplot(211)
        groups = df_all['group'].unique()
        data_by_group = [df_all[df_all['group'] == g]['duration_s'].values for g in groups]
        bp = ax1.boxplot(data_by_group, labels=groups, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Duration (s)', fontsize=12)
        ax1.set_title('Duration Comparison by Group', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Statistics table
        ax2 = fig.add_subplot(212)
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = []
        for group in stats.index:
            row = [group] + [f"{val:.2f}" if isinstance(val, float) else str(val) 
                           for val in stats.loc[group]]
            table_data.append(row)
        
        table = ax2.table(cellText=table_data,
                         colLabels=['Group', 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Total'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Save image button
        toolbar = ttk.Frame(window)
        toolbar.pack(fill='x', pady=5)
        def save_fig():
            fname = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")]
            )
            if fname:
                try:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image saved to:\n{fname}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        ttk.Button(toolbar, text="Save Image...", command=save_fig).pack(side='right')
        
        self.status_var.set("Group statistics displayed")
    
    def show_group_total(self):
        """Show total behavior duration comparison between groups"""
        df_all = self.load_data()
        if df_all is None:
            return
        
        window = tk.Toplevel(self.root)
        window.title("Group Total Duration")
        window.geometry("900x600")
        
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        totals = df_all.groupby('group')['duration_s'].sum().sort_values(ascending=False)
        sns.barplot(x=totals.index, y=totals.values, ax=ax, palette="Set2")
        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Total Duration (s)', fontsize=12)
        ax.set_title('Total Behavior Duration by Group', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate bars with values
        for i, v in enumerate(totals.values):
            ax.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=10)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Save image button
        toolbar = ttk.Frame(window)
        toolbar.pack(fill='x', pady=5)
        def save_fig():
            fname = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")]
            )
            if fname:
                try:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image saved to:\n{fname}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        ttk.Button(toolbar, text="Save Image...", command=save_fig).pack(side='right')
        
        self.status_var.set("Group total duration displayed")
        

    def export_report(self):
        """Export summary report to text file"""
        df_all = self.load_data()
        if df_all is None:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("   Batch Behavior Analysis Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files analyzed: {len(self.csv_files)}\n\n")
            
            # Group summary
            f.write("-"*60 + "\n")
            f.write("Group Summary\n")
            f.write("-"*60 + "\n")
            stats = df_all.groupby('group')['duration_s'].agg([
                ('Event_Count', 'count'),
                ('Mean_Duration', 'mean'),
                ('Median_Duration', 'median'),
                ('Std_Duration', 'std'),
                ('Total_Time', 'sum')
            ]).round(2)
            f.write(stats.to_string())
            f.write("\n\n")
            
            # Individual summary
            f.write("-"*60 + "\n")
            f.write("Individual Summary\n")
            f.write("-"*60 + "\n")
            ind_stats = df_all.groupby(['group', 'file_animal_id'])['duration_s'].agg([
                ('Events', 'count'),
                ('Mean_Dur', 'mean'),
                ('Total', 'sum')
            ]).round(2)
            f.write(ind_stats.to_string())
            f.write("\n\n")
            
            f.write("="*60 + "\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{filename}")
        self.status_var.set("Report exported")


def main():
    root = tk.Tk()
    app = BatchAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
