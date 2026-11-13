FROM ubuntu:24.04

# Install Python, R, and basic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    r-base \
    r-base-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy input data files into container
RUN mkdir -p "/app/data/pos_de_res"
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_28.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_300.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_3000.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_9.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_900.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_95.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Brefeldin-A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_28.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_300.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_300.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_3000.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_9.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_900.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_900.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_95.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Dexamethasone_95.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_28.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_300.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_3000.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_9.5.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_900.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_95.csv", "/app/data/pos_de_res/DE_results_human_aortic_smooth_muscle_cells_Trichostatin A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_28.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_300.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_3000.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_9.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_900.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_95.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Brefeldin-A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_28.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_300.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_300.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_3000.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_9.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_900.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_900.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_95.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Dexamethasone_95.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_28.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_300.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_3000.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_9.5.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_900.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_95.csv", "/app/data/pos_de_res/DE_results_human_dermal_fibroblast_Trichostatin A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_28.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_300.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_3000.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_9.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_900.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_95.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Brefeldin-A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_28.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_300.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_300.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_3000.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_9.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_900.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_900.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_95.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Dexamethasone_95.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_28.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_300.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_3000.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_9.5.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_900.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_95.csv", "/app/data/pos_de_res/DE_results_human_epithelial_melanocytes_Trichostatin A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_28.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_300.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_3000.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_9.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_900.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_95.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Brefeldin-A_95.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_28.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_300.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_300.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_3000.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_9.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_900.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_900.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_95.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Dexamethasone_95.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_28.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_28.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_300.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_300.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_3000.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_3000.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_9.5.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_9.5.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_900.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_900.csv"]
COPY ["data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_95.csv", "/app/data/pos_de_res/DE_results_human_skeletal_muscle_myoblasts_Trichostatin A_95.csv"]

WORKDIR /app
