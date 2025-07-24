import multiprocessing
import glob, os
import typer
import glob

from pathlib import Path                 
from eval_data_gen.core.knowledge.index_builder import build_index
from eval_data_gen.core.data_access.taxonomy_loader import load_taxonomy
from eval_data_gen.core.knowledge.bundle_builder import BundleBuilder
from eval_data_gen.core.generators.rag_generator import RAGGenerator
from eval_data_gen.core.data_access.taxonomy_manager import TaxonomyManager

app = typer.Typer()


@app.command()
def load_taxonomies_to_db(taxonomy_dir: str = "sample_data/taxonomies"):
    """to load all taxonomies from local files (.yaml/.json)
    into the MongoDB database. To seed the database."""

    typer.echo("Connecting to MongoDB and loading taxonomies...")
    try:
        manager = TaxonomyManager()
        manager.load_from_files_and_store(taxonomy_dir)
        typer.echo("Taxonomy loading complete.")
    except Exception as e:
        typer.echo(f"Error: {e}")
        typer.echo("Please ensure your MONGO_URI is set correctly in a .env file.")


@app.command()
def pipeline_run(taxonomy_dir: str = "sample_data/taxonomies", bundle_dir: str = "tmp/bundles", n: int = 3, k: int = 4):
    typer.echo("\nStep 1: Building FAISS Index...")
    build_index()

    # loading taxonomies from local file 
    # taxonomy_paths = sorted(Path(taxonomy_dir).glob("taxonomy_law_*.yaml"))
    # if not taxonomy_paths:
    #     typer.echo("No taxonomy_law_*.yaml files found.")
    #     raise typer.Exit()
    
    # print(taxonomy_paths)

    # for path in taxonomy_paths:
    #     print(path)
    #     typer.echo(f"\nStep 2: Processing taxonomy → {path}")
    #     builder = BundleBuilder(out_dir=bundle_dir, k=k, window=window)
    #     leaves = load_taxonomy(path)
    #     for leaf in leaves:
    #         builder.build(leaf)

    # load taxonomies from MongoDB
    typer.echo("\nStep 2: Loading taxonomies from MongoDB...")
    try:
        manager = TaxonomyManager()
        leaves = manager.get_all_leaves()
    except Exception as e:
        typer.echo(f"Error connecting to MongoDB: {e}")
        typer.echo("Please ensure your MONGO_URI is set and run `load-taxonomies-to-db` first.")
        raise typer.Exit()

    if not leaves:
        typer.echo("No taxonomies found in the database. Please run the `load-taxonomies-to-db` command first.")
        raise typer.Exit()
    
    typer.echo(f"Found {len(leaves)} taxonomy leaves in the database.")

    typer.echo("\nStep 3: Building bundles for each taxonomy leaf...")
    builder = BundleBuilder(out_dir=bundle_dir, k=k)
    for leaf in leaves:
        builder.build(leaf)


    typer.echo(f"\nStep 4: Generating MCQs for all bundles in {bundle_dir}")
    gen = RAGGenerator()
    bundle_paths = sorted(Path(bundle_dir).glob("*.json"))
    for bp in bundle_paths:
        gen.generate_for_bundle(bp, n=n)

    typer.echo("\nPipeline completed. Questions saved under tmp/questions/")

def main():
    # --- FIX FOR SEGMENTATION FAULT (MUST BE HERE) ---
    # This sets the multiprocessing start method to 'spawn', which is safer on macOS.
    # It must be called before any other multiprocessing-related code runs.
    try:
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method("spawn", force=True)
            print("--- Multiprocessing start method set to 'spawn' to prevent crashes ---")
    except RuntimeError:
        # This can happen if the context is already set. It's safe to ignore.
        pass
    
    app()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     app()