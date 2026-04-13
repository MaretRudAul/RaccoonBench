from pathlib import Path


class Loader:
    """
    A class that represents a loader for OpenAI's GPTs.

    Args:
        gpts_dir (str): The directory path where the GPTs are located.

    Each ``for ... in loader`` or ``list(loader)`` starts a **new** directory listing.
    This matters when ``RaccoonGang.benchmark()`` is invoked multiple times (e.g.
    ``--run_all_three_benchmark_modes``): a single-shot iterator would be exhausted
    after the first pass and later modes would run over zero GPTs (empty ``runs``).
    """

    def __init__(self, gpts_dir: str) -> None:
        self.gpts_dir = Path(gpts_dir)

    def __iter__(self):
        return iter(self.gpts_dir.glob("*"))

class AttLoader:
    def __init__(self, att_dir: str) -> None:
        self.att_categories_path = list(Path(att_dir).glob("*"))
        self.category_names = [category_path.name for category_path in self.att_categories_path]

        self.prompt_paths = {}
        for category_name, category_path in zip(self.category_names,list(self.att_categories_path)):
            self.prompt_paths[category_name] = list(category_path.glob("*"))
