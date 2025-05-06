#!/usr/bin/env python3
import os
import re
from pathlib import Path

results_dir = Path('results')

# First pass: handle the mistral-thinking replacements and epoch2 removals
for file_path in results_dir.iterdir():
    if file_path.is_file():
        old_name = file_path.name
        
        # Handle any files with mistral-thinking in the name
        if 'mistral-thinking' in old_name:
            new_name = old_name.replace('mistral-thinking', 'nlistral')
            
            # Remove epochs2 or epoch2 from filename
            new_name = new_name.replace('-epochs2-', '-')
            new_name = new_name.replace('epochs2', '')
            new_name = new_name.replace('-epoch2-', '-')
            new_name = new_name.replace('epoch2', '')
            
            # Avoid duplicate hyphens
            new_name = re.sub(r'-+', '-', new_name)
            
            # Check if we made any changes
            if new_name != old_name:
                old_path = results_dir / old_name
                new_path = results_dir / new_name
                
                print(f"Renaming: {old_name} -> {new_name}")
                os.rename(old_path, new_path)
        
        # Also handle any files with just "mistral-" but not "mistral-thinking"
        elif old_name.startswith('mistral-') and not old_name.startswith('mistral-base'):
            new_name = old_name.replace('mistral-', 'nlistral-')
            
            # Remove epochs2 or epoch2 from filename
            new_name = new_name.replace('-epochs2-', '-')
            new_name = new_name.replace('epochs2', '')
            new_name = new_name.replace('-epoch2-', '-')
            new_name = new_name.replace('epoch2', '')
            
            # Avoid duplicate hyphens
            new_name = re.sub(r'-+', '-', new_name)
            
            # Check if we made any changes
            if new_name != old_name:
                old_path = results_dir / old_name
                new_path = results_dir / new_name
                
                print(f"Renaming: {old_name} -> {new_name}")
                os.rename(old_path, new_path)
        
        # Process all nlistral files with epoch2 still in name
        elif 'nlistral-' in old_name and 'epoch2' in old_name:
            new_name = old_name.replace('-epoch2-', '-')
            new_name = new_name.replace('epoch2', '')
            
            # Avoid duplicate hyphens
            new_name = re.sub(r'-+', '-', new_name)
            
            # Check if we made any changes
            if new_name != old_name:
                old_path = results_dir / old_name
                new_path = results_dir / new_name
                
                print(f"Renaming: {old_name} -> {new_name}")
                os.rename(old_path, new_path)

# Second pass: replace ablation patterns
for file_path in results_dir.iterdir():
    if file_path.is_file():
        old_name = file_path.name
        
        # Replace 'ablation-X-best' with 'ablationX'
        if 'nlistral-ablation-' in old_name:
            # Replace ablation-X-best with ablationX
            if 'ablation-21-best' in old_name:
                new_name = old_name.replace('ablation-21-best', 'ablation21')
            elif 'ablation-22-best' in old_name:
                new_name = old_name.replace('ablation-22-best', 'ablation22')
            elif 'ablation-0-best' in old_name:
                new_name = old_name.replace('ablation-0-best', 'ablation0')
            elif 'ablation-1-best' in old_name:
                new_name = old_name.replace('ablation-1-best', 'ablation1')
            elif 'ablation-2-best' in old_name:
                new_name = old_name.replace('ablation-2-best', 'ablation2')
            elif 'ablation-3-best' in old_name:
                new_name = old_name.replace('ablation-3-best', 'ablation3')
            else:
                new_name = old_name
                
            # Check if we made any changes
            if new_name != old_name:
                old_path = results_dir / old_name
                new_path = results_dir / new_name
                
                print(f"Renaming: {old_name} -> {new_name}")
                os.rename(old_path, new_path)

# Convert ablX to ablationX patterns
for file_path in results_dir.iterdir():
    if file_path.is_file():
        old_name = file_path.name
        
        # Handle abl patterns in nlistral files
        if 'nlistral-abl' in old_name:
            # Special cases with underscores first
            if 'abl3_1' in old_name:
                new_name = old_name.replace('abl3_1', 'ablation31')
            elif 'abl3_2' in old_name:
                new_name = old_name.replace('abl3_2', 'ablation32')
            elif 'abl2_1' in old_name:
                new_name = old_name.replace('abl2_1', 'ablation21')
            elif 'abl2_2' in old_name:
                new_name = old_name.replace('abl2_2', 'ablation22')
            # Standard ablation numbers
            elif 'abl0' in old_name:
                new_name = old_name.replace('abl0', 'ablation0')
            elif 'abl1' in old_name:
                new_name = old_name.replace('abl1', 'ablation1')
            elif 'abl2' in old_name:
                new_name = old_name.replace('abl2', 'ablation2')
            elif 'abl3' in old_name:
                new_name = old_name.replace('abl3', 'ablation3')
            else:
                new_name = old_name
            
            # Check if we made any changes
            if new_name != old_name:
                old_path = results_dir / old_name
                new_path = results_dir / new_name
                
                print(f"Renaming: {old_name} -> {new_name}")
                os.rename(old_path, new_path)

print("Renaming complete!") 