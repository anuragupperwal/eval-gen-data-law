import multiprocessing
import typer
import dask
from dask.distributed import Client
from pathlib import Path                 
from tqdm import tqdm

from eval_data_gen.core.knowledge.index_builder import build_index
from eval_data_gen.core.knowledge.bundle_builder import BundleBuilder
from eval_data_gen.core.generators.rag_generator import RAGGenerator
from eval_data_gen.core.data_access.taxonomy_manager import TaxonomyManager

app = typer.Typer()


# dask worker func
def process_single_bundle(bundle_path_str: str, questions_out_dir_str: str, n: int):
    """
    Worker function to be parallelized by Dask.
    Processes one bundle and writes one JSON file.
    It's self-contained to avoid sharing state between processes.
    This function encapsulates the work for a single bundle. Dask will run this function in parallel for each bundle file.
    """
    try:
        bundle_path = Path(bundle_path_str)
        questions_out_dir = Path(questions_out_dir_str)
        output_filename = questions_out_dir / bundle_path.name

        # Skip if the output file already exists.
        if output_filename.exists():
            return f"Skipped: {bundle_path.stem}"

        # Each Dask worker creates its own RAGGenerator instance.
        gen = RAGGenerator()
        gen.generate_for_bundle(bundle_path, n=n)
        return f"Success: {bundle_path.stem}"
    except Exception as e:
        return f"Failed: {bundle_path.stem} with error: {e}"



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
def pipeline_run(bundle_dir: str = "tmp/bundles", questions_out_dir: str = "tmp/questions", n: int = 3, k: int = 4):
    typer.echo("\nStep 1: Building FAISS Index...")
    build_index()

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
    for leaf in tqdm(leaves, desc="Building bundles"):
        builder.build(leaf) #build bundles


    typer.echo(f"\nStep 4: Generating MCQs for all bundles in {bundle_dir} using DASK")
    bundle_paths = sorted(Path(bundle_dir).glob("*.json"))
    #to ensure output dir
    Path(questions_out_dir).mkdir(parents=True, exist_ok=True) 

    # init DASK client for local parallel processing
    client = Client()
    typer.echo(f"Dask dashboard available at: {client.dashboard_link}")
    
    #create lazy tasks
    lazy_tasks = []
    for bp in bundle_paths:
        # RAGGenerator() called in here
        task = dask.delayed(process_single_bundle)(str(bp), questions_out_dir, n)
        lazy_tasks.append(task)
            


    if not lazy_tasks:
        typer.echo("No bundles found to process. Exiting generation step.")
    else:
        typer.echo(f"Executing {len(lazy_tasks)} generation tasks in parallel...")
        
        # Compute all tasks, distributed across all CPU cores.
        results = dask.compute(*lazy_tasks)
        
        # Optional: Summarize the results from the parallel run
        success_count = sum(1 for r in results if r.startswith("Success"))
        skipped_count = sum(1 for r in results if r.startswith("Skipped"))
        failed_count = len(results) - success_count - skipped_count
        
        typer.echo(f"Successfully generated: {success_count}")
        typer.echo(f"Skipped (already exist): {skipped_count}")
        typer.echo(f"Failed: {failed_count}")


    typer.echo("\nPipeline completed. Questions saved under tmp/questions/")

def main():
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






