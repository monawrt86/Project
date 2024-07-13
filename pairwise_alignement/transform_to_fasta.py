def transform_to_fasta(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    fasta_data = []
    current_header = None
    current_sequences = {}
    protein_name = ""
    keep_section = False

    for line in lines:
        if line.startswith('>'):
            # Check if the current section should be kept
            if 'HOMSTRAD' in line:
                keep_section = True
                # Write the previous header and sequences if present
                if current_header:
                    fasta_data.append(f">{protein_name} {current_header}\n")
                    for key in current_sequences:
                        formatted_sequence = format_sequence(current_sequences[key])
                        fasta_data.append(f"{key}\n{formatted_sequence}\n")
                    fasta_data.append("\n")  # Add a blank line between sections
                # Initialize for new header and sequences
                parts = line.strip().split(maxsplit=1)
                protein_name = parts[0][1:] if len(parts) > 0 else ""
                current_header = parts[1] if len(parts) > 1 else ""
                current_sequences = {}
            else:
                keep_section = False
        elif line.strip() and keep_section:
            parts = line.split()
            seq_id = parts[0]
            sequence = ''.join(parts[1:])
            sequence = ''.join([c for c in sequence if not c.isdigit()])  # Remove numbers
            if seq_id not in current_sequences:
                current_sequences[seq_id] = sequence
            else:
                current_sequences[seq_id] += sequence
    
    # Write the last header and sequences
    if keep_section and current_header:
        fasta_data.append(f">{protein_name} {current_header}\n")
        for key in current_sequences:
            formatted_sequence = format_sequence(current_sequences[key])
            fasta_data.append(f"{key}\n{formatted_sequence}\n")
        fasta_data.append("\n")  # Add a blank line at the end of the file
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(fasta_data)

def format_sequence(sequence, line_length=80):
    return '\n'.join([sequence[i:i+line_length] for i in range(0, len(sequence), line_length)])

# Usage
transform_to_fasta('C:\\Users\\wrtmo\\Documents\\Internship\\pairwise_alignement\\data\\homstrad_data\\homstrad_alignments.txt', 'output.fasta')
