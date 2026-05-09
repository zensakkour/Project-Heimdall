with open("src/docs/RESEARCH_PAPER.md", "r") as f:
    content = f.read()

# I will append a short section at the end of the paper describing the new Operator Dashboard

append = """
## 9. Operator Dashboard and Local Visualization
Beyond benchmarking, a crucial aspect of Heimdall's development was providing a transparent, local-first interface for analyzing individual images. Early versions of the interface provided basic candidate visualization, but the requirements evolved to necessitate a "serious visual investigation console" known as Heimdall Operator Mode.

The operator mode (`/api/operator/*`) introduces an isolated session architecture. It executes the Heimdall pipeline chronologically, capturing stage transitions, explicit warnings if candidate providers fail or degrade, and structural clues from the RF-DETR detections. To support transparent human-in-the-loop interaction, the operator UI renders the fused probability estimate alongside the generated evidence, and records user-added notes and pins. This interface explicitly replaces "silent failures" with verbose failure states and timeline reporting, establishing the framework for future multi-operator and workflow-oriented investigations.
"""

content += append

with open("src/docs/RESEARCH_PAPER.md", "w") as f:
    f.write(content)
