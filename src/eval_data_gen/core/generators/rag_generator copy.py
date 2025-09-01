import json, re, time
from pathlib import Path
from eval_data_gen.core.providers.gemini_provider import GeminiProvider
from eval_data_gen.core.providers.perplexity_sonar_provider import PerplexitySonarProvider
from eval_data_gen.core.data_access.prompt_manager import PromptManager


# _QA_RE = re.compile(r"Q\d+\..+?Answer:\s*[A-F0-6]", re.S)

# PROMPT_TMPL = """You are a law-domain question setter.

# Goal: Using ONLY the passages below, write {n} {diff_tag}-level multiple-choice questions (1-6) with exactly one correct answer per question. And add two more options 7. None of the above, 8. Question not clear to every question.
# Every question MUST be fully answerable without seeing the passages. That means:
#   • No references like "According to the passage", "Based on the excerpt", "As stated above", etc.
#   • If you mention a statute/section/case, include the essential meaning/definition in the stem itself. But do mention section/case/statute along with the meaning.
#     Example (GOOD): "Section 556 of the Code of Criminal Procedure, 1898, disqualifies a Magistrate from trying a case if they have a personal interest. Which of the following situations triggers this disqualification?"
#     Example (BAD):  "According to Section 556 discussed in the passage, which situation triggers disqualification?"
#   • Avoid vague pronouns (“this”, “that”, “it”, “they”) without a clear antecedent.
#   • Each distractor must be plausible but clearly wrong if the reader understands the concept.
#   • Use only facts, terms, and relationships present in the passages. Do not invent outside information.

# Write questions that test understanding of *{leaf_label}*, not rote fact copying. Prefer concept/application questions over direct quotation.

# Passages (context you MUST rely on, but MUST NOT reference explicitly in the question text):
# ---------
# {context}
# ---------

# STRICT OUTPUT FORMAT (no extra text, no explanations):
# Q1. <question text>
# (1) option A
# (2) option B
# (3) option C
# (4) option D
# (5) option E
# (6) option F
# Answer: X

# """


class RAGGenerator:
    def __init__(self, out_dir="tmp/questions"):
        self.provider1 = GeminiProvider()
        self.provider2 = PerplexitySonarProvider()
        self.out_dir  = Path(out_dir); 
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_manager = PromptManager()

    def _parse(self, text: str, parser_regex: str):
        # return _QA_RE.findall(text)
        qa_re = re.compile(parser_regex, re.S)
        return qa_re.findall(text)

    def generate_for_bundle(        
        self,
        bundle_path: Path,
        prompt_id: str,
        options: int,
        n: int = 3
    ):
        
        # Fetch the prompt data (template and parser) from MongoDB
        try:
            prompt_data = self.prompt_manager.get_prompt(prompt_id)
            prompt_template = prompt_data["template"]
            parser_regex = prompt_data["parser_regex"]
        except Exception as e:
            print(f"Error fetching prompt '{prompt_id}': {e}")
            return

        # load bundles 
        bundle = json.loads(bundle_path.read_text())
        context = "\n\n".join(p["text"] for p in bundle["passages"])
        difficulty = bundle.get("difficulty", "H")        
        prompt = prompt_template.format(
            n=n,
            diff_tag=difficulty,                          
            leaf_label=bundle["taxonomy_id"].split(".")[-1],
            context=context,
            options=options,
        )

        time.sleep(15)
        raw = self.provider1.generate(prompt)
        qas = self._parse(raw, parser_regex)

        out = {"taxonomy_id": bundle["taxonomy_id"], "mcqs": qas}
        out_file = self.out_dir / f"{bundle['taxonomy_id']}.json"
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"Gen- {bundle['taxonomy_id']}")





