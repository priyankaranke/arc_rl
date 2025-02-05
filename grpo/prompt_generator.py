"""Module for prompt generation."""

PREFIX = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz"
INP_PREFIX = "I"
OUT_PREFIX = "O"
ARR_SEP = "\n"
ARR_END = "\n"
PRE_OUT = "+/-="
EXA_END = "</s>"

def convert_batch_to_prompts(
    batch: list,
    tokenizer,
    max_seq_length: int,
    generation_buffer: int,
):
    """
    Convert a batch of leave-one-out examples into prompts.
    """
    # This was used in the finetuning process so we use the same format here.
    prefix = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz"
    inp_prefix = "I"
    out_prefix = "O"
    arr_sep = "\n"
    arr_end = "\n"
    pre_out = "+/-="
    exa_end = "</s>"

    def fmt_2d_array(t2d):
        """
        Convert a 2D torch.Tensor or list-of-lists into a string with line breaks.
        E.g. [[4,9,7],[2,6,7]] -> '497\n267\n'
        """
        if hasattr(t2d, "tolist"):
            t2d = t2d.tolist()
        lines = []
        for row in t2d:
            # row is e.g. [4,9,7]; convert to string '497'
            row_str = "".join(str(x) for x in row)
            lines.append(row_str)
        return arr_sep.join(lines) + arr_end

    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    max_prompt_length = max_seq_length - generation_buffer
    prompts = []

    for example in batch:
        prompt_text = ""
        current_token_count = 0

        withheld_in = fmt_2d_array(example["withheld_input"])
        withheld_puzzle = f"{inp_prefix}{withheld_in}{pre_out}{out_prefix}"
        withheld_puzzle_len = token_len(withheld_puzzle)

        def make_demo_segment(demo_in_2d, demo_out_2d, is_first=False):
            seg = ""
            if is_first:
                seg += prefix
            seg += inp_prefix
            seg += fmt_2d_array(demo_in_2d)
            seg += pre_out
            seg += out_prefix
            seg += fmt_2d_array(demo_out_2d)
            seg += exa_end
            return seg

        first_demo = True
        for demo_in, demo_out in example.get("other_demos", []):
            demo_segment = make_demo_segment(demo_in, demo_out, is_first=first_demo)
            demo_segment_len = token_len(demo_segment)

            # Check if we can add this demo + the withheld puzzle within max_prompt_length
            if (
                current_token_count + demo_segment_len + withheld_puzzle_len
                <= max_prompt_length
            ):
                prompt_text += demo_segment
                current_token_count += demo_segment_len
                first_demo = False
            else:
                # skip this demo; no space
                pass

        prompt_text += withheld_puzzle
        current_token_count += withheld_puzzle_len

        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(token_ids) > max_prompt_length:
            # Keep last max_prompt_length tokens
            token_ids = token_ids[-max_prompt_length:]
            prompt_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        prompts.append(prompt_text)

    return prompts
