import json

from scene_search.ollama_client import OllamaClient


def rerank_results(config: dict, query: str, results: list) -> list:
    client = OllamaClient(config["ollama_base_url"])
    model = client.choose_text_model(config.get("ollama_text_model"))
    if not model:
        return results
    payload = [{"candidate_id": result.scene_id, "caption": result.caption, "start_time": result.start_time, "end_time": result.end_time} for result in results]
    prompt = f"Given a user query and candidate scene captions, rank the candidates by relevance. Return only a JSON list of objects with candidate_id and relevance_score from most relevant to least relevant.\nUser query: {query}\nCandidates:\n{json.dumps(payload)}"
    try:
        response = client.generate(model, prompt)
        order = parse_rerank_response(response)
        mapping = {result.scene_id: result for result in results}
        reranked = []
        for item in order:
            candidate_id = item.get("candidate_id") if isinstance(item, dict) else str(item)
            original = mapping.get(candidate_id)
            if original:
                original.original_rank = original.rank
                original.original_score = original.score
                if isinstance(item, dict) and "relevance_score" in item:
                    original.score = float(item["relevance_score"])
                reranked.append(original)
        remaining = [result for result in results if result not in reranked]
        combined = reranked + remaining
        for index, result in enumerate(combined, start=1):
            result.rank = index
        return combined
    except Exception:
        return results


def parse_rerank_response(response: str):
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return json.loads(text)
