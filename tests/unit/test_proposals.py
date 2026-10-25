import dataclasses

import pytest

from forgemind.core.proposal import (
    ArchitectureProposal,
    AttackTestProposal,
    Candidate,
    CodePatchProposal,
)


def test_proposal_validation_empty_patch_rejected():
    with pytest.raises(ValueError):
        CodePatchProposal(agent_role="builder")


def test_proposal_created_and_modified_overlap_rejected():
    with pytest.raises(ValueError):
        CodePatchProposal(created_files={"a.py": "x"}, modified_files={"a.py": "y"})


def test_architecture_requires_components():
    with pytest.raises(ValueError):
        ArchitectureProposal(agent_role="architect")


def test_attack_test_requires_name_and_code():
    with pytest.raises(ValueError):
        AttackTestProposal(agent_role="critic", test_name="t", test_code="")
    with pytest.raises(ValueError):
        AttackTestProposal(agent_role="critic", test_name="", test_code="pass")


def test_candidate_proposal_required_m10():
    # M10: the old proposal default factory built a no-file patch that
    # CodePatchProposal.__post_init__ rejects, so this raised ValueError
    # from inside the factory. Now the missing field is a TypeError at
    # the call site.
    with pytest.raises(TypeError):
        Candidate(parent_node_id="root")
    with pytest.raises(TypeError):
        Candidate()


def test_candidate_requires_parent():
    patch = CodePatchProposal(created_files={"a.py": "x"})
    with pytest.raises(ValueError):
        Candidate(proposal=patch)
    c = Candidate(parent_node_id="root", proposal=patch)
    assert c.parent_node_id == "root"


def test_proposals_immutable():
    p = CodePatchProposal(created_files={"a.py": "x"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.created_files = {}
