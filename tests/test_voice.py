"""Tests for voice and natural-language control.

The class that matters is TestRefusals. Everything else here is about
convenience; that one is about a microphone being an unauthenticated
control surface in a room that may contain bystanders, two-way radios and
the show's own audio track playing the word "start".

Note what is NOT tested: the language-model branch of `compose`. There is
no API key in CI, and a test that needs the network fails for reasons
unrelated to the code. The rule-based composer is the default and is what
runs here; both paths are held to the same contract by `_sanitise_spec`,
which is tested directly against deliberately hostile input.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from skyshine import voice


def intent(text, **ctx):
    return voice.parse(text, ctx or {"selected": 0})


# ==========================================================================
# Refusals
# ==========================================================================


class TestRefusals:
    """Arming, starting and disarming must be unreachable by voice."""

    @pytest.mark.parametrize("phrase", [
        "arm", "arm the fleet", "arm everything", "arming now",
        "start", "start the show", "start it", "launch", "launch the show",
        "take off", "takeoff", "take off now", "begin the show",
        "go for show", "disarm", "disarm the fleet", "dis arm",
    ])
    def test_flight_critical_phrases_are_refused(self, phrase):
        i = intent(phrase)
        assert i.risk == voice.REFUSED, f"{phrase!r} parsed as {i.risk}"
        assert i.name in voice.REFUSED_INTENTS
        assert "not available by voice" in i.speech

    @pytest.mark.parametrize("phrase", [
        "please arm the drones now",
        "ok everyone, start the show",
        "I think we should launch",
        "can you arm them",
    ])
    def test_refused_even_when_politely_embedded(self, phrase):
        assert intent(phrase).risk == voice.REFUSED

    def test_refusal_beats_every_other_rule(self):
        """A phrase containing both must refuse, not do the safe half."""
        for phrase in ("validate and then arm the fleet",
                       "prepare the show and start it",
                       "add a heart then launch"):
            assert intent(phrase).risk == voice.REFUSED, phrase

    def test_refused_intents_carry_no_arguments(self):
        for phrase in ("arm", "start the show", "disarm"):
            assert intent(phrase).args == {}

    def test_the_refused_set_is_exactly_these_three(self):
        # A grammar change must not quietly widen what voice can do.
        assert voice.REFUSED_INTENTS == {"arm", "start", "disarm"}

    def test_server_refuses_independently_of_the_parser(self, client):
        """Two independent gates for the three dangerous verbs."""
        r = client.post("/api/voice", json={
            "text": "arm the fleet", "context": {}, "confirmed": True,
        }).json()
        assert r["refused"] is True
        assert r["executed"] is False

    def test_confirmed_true_cannot_launder_a_refusal(self, client):
        """Even a client claiming the operator confirmed gets nothing."""
        for phrase in ("start the show", "arm", "disarm"):
            r = client.post("/api/voice", json={
                "text": phrase, "context": {}, "confirmed": True,
            }).json()
            assert r.get("refused") is True, phrase
            assert r["executed"] is False


# ==========================================================================
# Risk classification
# ==========================================================================


class TestRisk:
    @pytest.mark.parametrize("phrase,name", [
        ("suspend", "suspend"),
        ("freeze", "suspend"),
        ("hold position", "suspend"),
        ("all hold", "suspend"),
    ])
    def test_stopping_motion_is_immediate(self, phrase, name):
        """Suspend reduces energy, so it runs without a confirm step.

        The fleet stops where it is and keeps whatever separation it had.
        Making the operator click first would add latency to the one
        command that is always safe to obey.
        """
        i = intent(phrase)
        assert i.name == name
        assert i.risk == voice.AUTO

    @pytest.mark.parametrize("phrase", [
        "resume", "continue the show",
        "land", "land now", "land the fleet",
        "collective return home", "bring everyone home", "all return home",
        "upload the plans", "adapt to 50 drones",
        "load the fleet on the simulator",
    ])
    def test_committing_aircraft_to_motion_needs_a_click(self, phrase):
        assert intent(phrase).risk == voice.CONFIRM, phrase

    @pytest.mark.parametrize("phrase", [
        "prepare", "validate", "run preflight", "auto geofence",
        "export vviz", "top view", "play", "status", "help",
    ])
    def test_design_and_queries_run_immediately(self, phrase):
        assert intent(phrase).risk == voice.AUTO, phrase

    def test_confirm_intents_carry_a_prompt(self):
        """A confirm card has to actually ask something, not just narrate."""
        for phrase in ("land now", "collective return home", "upload the plans",
                       "resume", "adapt to 50 drones"):
            i = intent(phrase)
            assert i.confirm_prompt, f"{phrase!r} has no confirm prompt"
            assert "?" in i.confirm_prompt, f"{phrase!r} prompt asks nothing"

    def test_every_intent_has_a_known_risk(self):
        for example in voice.vocabulary()["examples"]:
            assert intent(example).risk in (voice.AUTO, voice.CONFIRM, voice.REFUSED)


# ==========================================================================
# Parsing
# ==========================================================================


class TestNumbers:
    @pytest.mark.parametrize("spoken,expected", [
        ("one", 1), ("ten", 10), ("twenty", 20), ("fifty", 50),
        ("one hundred", 100), ("two hundred", 200), ("five hundred", 500),
        ("a hundred and fifty", 150), ("one hundred and twenty eight", 128),
        ("three hundred and sixty", 360), ("one thousand", 1000),
    ])
    def test_spoken_numbers_become_digits(self, spoken, expected):
        assert str(expected) in voice.words_to_number(spoken)

    def test_digits_pass_through(self):
        assert voice.words_to_number("add 250 drones") == "add 250 drones"

    def test_mixed_text_keeps_its_words(self):
        out = voice.words_to_number("create a show with two hundred drones")
        assert "200" in out and "drones" in out and "create" in out

    def test_number_extraction_after_a_keyword(self):
        assert voice.find_number("adapt to 180 drones", after="to") == 180
        assert voice.find_number("set the tempo to 128", after="tempo") == 128
        assert voice.find_number("nothing here", after="to") is None


class TestVocabularyLookup:
    @pytest.mark.parametrize("word,shape", [
        ("sphere", "ball"), ("globe", "ball"), ("orb", "ball"),
        ("heart", "heart"), ("star", "star"), ("cube", "cube"),
        ("box", "cube"), ("donut", "torus"), ("dna", "double_helix"),
        ("spiral", "helix"), ("disk", "disc"),
    ])
    def test_spoken_shape_words(self, word, shape):
        found = voice.find_shape(f"add a {word}")
        assert found and found[0] == shape

    def test_longest_phrase_wins(self):
        """'double helix' must not match the bare 'helix' rule first."""
        found = voice.find_shape("add a double helix")
        assert found and found[0] == "double_helix"

    def test_misheard_shape_is_recovered_fuzzily(self):
        found = voice.find_shape("add a sphear")
        assert found and found[0] == "ball"
        assert found[2] < 1.0, "a fuzzy match must report lower confidence"

    def test_every_shape_word_maps_to_a_real_generator(self):
        from skyshine import geometry

        for word, (gen, _params) in voice.SHAPE_WORDS.items():
            assert gen in geometry.SHAPES, f"{word!r} -> {gen!r} does not exist"

    def test_every_effect_word_maps_to_a_real_effect(self):
        from skyshine import effects

        for word, name in voice.EFFECT_WORDS.items():
            assert name in effects.EFFECTS, f"{word!r} -> {name!r} does not exist"

    @pytest.mark.parametrize("word", ["red", "blue", "cyan", "gold", "warm white"])
    def test_colour_words(self, word):
        assert voice.find_colour(f"make it {word}") == voice.COLOUR_WORDS[word]

    def test_colours_are_valid_rgb(self):
        for name, rgb in voice.COLOUR_WORDS.items():
            assert len(rgb) == 3, name
            assert all(0 <= c <= 255 for c in rgb), name


class TestIntents:
    def test_create_show_with_a_name_and_count(self):
        i = intent("create a show called summer gala with two hundred drones")
        assert i.name == "create_show"
        assert i.args["drone_count"] == 200
        assert i.args["name"] == "summer gala"

    def test_create_show_defaults(self):
        i = intent("new show")
        assert i.name == "create_show"
        assert i.args["drone_count"] == 100

    def test_add_formation_with_colour_and_altitude(self):
        i = intent("add a red heart at 60 metres")
        assert i.name == "add_formation"
        assert i.args["shape"] == "heart"
        assert i.args["color"] == list(voice.COLOUR_WORDS["red"])
        assert i.args["altitude"] == 60.0

    def test_heart_and_star_default_to_filled(self):
        """Spoken shapes mean the version that can actually fly."""
        for word in ("heart", "star"):
            i = intent(f"add a {word}")
            assert i.args["params"].get("solid") is True, word

    def test_add_formation_without_a_shape_asks_which(self):
        i = intent("add a formation")
        assert i.name == "unknown"
        assert "sphere" in i.speech

    def test_effect_targets_the_selected_formation(self):
        i = voice.parse("apply a chase effect", {"selected": 3})
        assert i.name == "apply_effect"
        assert i.args["effect"] == "chase"
        assert i.args["formation"] == 3

    def test_effect_picks_up_a_colour(self):
        i = intent("apply a chase effect in cyan")
        assert i.args["params"]["color"] == list(voice.COLOUR_WORDS["cyan"])

    def test_strobe_rate(self):
        i = intent("strobe at 4 hertz")
        assert i.args["effect"] == "strobe"
        assert i.args["params"]["hz"] == 4.0

    def test_gradient_colour_goes_to_the_start_parameter(self):
        """gradient takes start/end, not color - a wrong key is a TypeError."""
        i = intent("apply a gradient in blue")
        assert "start" in i.args["params"]

    def test_make_it_colour_becomes_a_solid_effect(self):
        i = voice.parse("make it blue", {"selected": 2})
        assert i.name == "apply_effect"
        assert i.args["effect"] == "solid"
        assert i.args["formation"] == 2

    def test_tempo(self):
        i = intent("set the tempo to one hundred and twenty eight")
        assert i.name == "set_music"
        assert i.args["bpm"] == 128.0

    def test_seek(self):
        assert intent("go to 45 seconds").args["seconds"] == 45.0

    @pytest.mark.parametrize("phrase,view", [
        ("top view", "top"), ("from the top", "top"),
        ("crowd view", "crowd"), ("audience view", "crowd"),
        ("show the map", "map"), ("iso", "iso"),
    ])
    def test_camera(self, phrase, view):
        assert intent(phrase).args["view"] == view

    def test_export_needs_a_format(self):
        assert intent("export vviz").args["format"] == "vviz"
        assert intent("export csv").args["format"] == "csv"
        assert intent("export").name == "unknown"

    def test_unknown_phrase_is_handled_not_crashed(self):
        i = intent("banana flavoured elephants")
        assert i.name == "unknown"
        assert i.confidence == 0.0
        assert "help" in i.speech

    def test_empty_input(self):
        i = intent("")
        assert i.name == "none"
        assert i.confidence == 0.0

    def test_transcript_is_preserved_verbatim(self):
        raw = "Add a RED heart, please!"
        assert voice.parse(raw, {}).transcript == raw

    def test_common_mishearings_are_corrected(self):
        assert intent("run pre flight").name == "preflight"
        assert intent("draw the geo fence").name == "auto_geofence"
        assert intent("export v viz").args["format"] == "vviz"


# ==========================================================================
# Composer
# ==========================================================================


class TestCompose:
    def test_rule_composer_is_the_default_without_a_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        spec = voice.compose("a calm opening", 80)
        assert spec["source"] == "rules"
        assert spec["drone_count"] == 80

    @pytest.mark.parametrize("brief,mood", [
        ("something romantic for a wedding", "romantic"),
        ("high energy new year party", "festive"),
        ("calm ambient opening", "calm"),
        ("a flag show for independence day", "patriotic"),
        ("fast punchy rock drop", "energetic"),
    ])
    def test_mood_detection(self, brief, mood, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert voice.compose(brief, 50)["mood"] == mood

    def test_named_shapes_appear_first_and_in_order(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        spec = voice.compose("a show with a heart and then a star", 60)
        shapes = [f["shape"] for f in spec["formations"]]
        assert shapes[0] == "heart"
        assert shapes[1] == "star"

    def test_output_is_always_buildable(self, monkeypatch):
        """Every composed formation must name a real shape and effect."""
        from skyshine import effects, geometry

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        for brief in ("anything", "a wedding", "diwali celebration",
                      "", "!!!", "a dragon breathing fire",
                      "something slow", "a patriotic flag show"):
            spec = voice.compose(brief, 60)
            assert spec["formations"]
            for f in spec["formations"]:
                assert f["shape"] in geometry.SHAPES
                assert f["effect"] in effects.EFFECTS
                # Raises if a parameter is missing: fade and wipe once
                # arrived without their duration and were dropped.
                effects.make(f["effect"], **f["effect_params"])
                assert len(f["color"]) == 3
                assert 20 <= f["altitude"] <= 110
                assert 4 <= f["transition"] <= 25
                assert 1 <= f["hold"] <= 20

    def test_bpm_is_picked_out_of_the_brief(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert voice.compose("a party at 128 bpm", 40)["bpm"] == 128.0

    def test_draft_says_it_is_not_flyable(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert "not flyable" in voice.compose("anything", 40)["note"].lower()


class TestSanitiser:
    """The contract both composers are held to, hostile input included."""

    def test_unknown_shape_is_corrected_or_replaced(self):
        out = voice._sanitise_spec({"formations": [
            {"shape": "sphereoid", "effect": "solid", "color": [1, 2, 3]},
        ]}, 50)
        from skyshine import geometry

        assert out["formations"][0]["shape"] in geometry.SHAPES
        assert out["corrections"]

    def test_misspelt_names_are_fuzzily_recovered(self):
        out = voice._sanitise_spec({"formations": [
            {"shape": "hart", "effect": "strob", "color": [1, 2, 3]},
        ]}, 50)
        assert out["formations"][0]["shape"] == "heart"
        assert out["formations"][0]["effect"] == "strobe"

    def test_out_of_range_numbers_are_clamped(self):
        out = voice._sanitise_spec({"formations": [
            {"shape": "ball", "effect": "solid", "color": [0, 0, 0],
             "altitude": 5000, "transition": 900, "hold": -3},
        ]}, 50)
        f = out["formations"][0]
        assert f["altitude"] == 110.0
        assert f["transition"] == 25.0
        assert f["hold"] == 1.0

    @pytest.mark.parametrize("colour", ["red", None, [300, -5, 10], [1, 2], {}, "#fff"])
    def test_bad_colours_never_reach_the_generator(self, colour):
        out = voice._sanitise_spec({"formations": [
            {"shape": "ball", "effect": "solid", "color": colour},
        ]}, 50)
        rgb = out["formations"][0]["color"]
        assert len(rgb) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)

    def test_missing_fields_get_defaults(self):
        out = voice._sanitise_spec({"formations": [{"shape": "ball"}]}, 50)
        f = out["formations"][0]
        assert f["effect"] in ("solid",)
        assert f["transition"] > 0 and f["hold"] > 0

    def test_empty_spec_still_produces_something_buildable(self):
        out = voice._sanitise_spec({"formations": []}, 50)
        assert len(out["formations"]) == 1
        assert "no usable formations" in " ".join(out["corrections"])

    def test_absurdly_long_spec_is_capped(self):
        out = voice._sanitise_spec(
            {"formations": [{"shape": "ball"} for _ in range(50)]}, 50)
        assert len(out["formations"]) <= 8

    def test_corrections_are_reported_not_hidden(self):
        out = voice._sanitise_spec({"formations": [
            {"shape": "nonsense", "effect": "nonsense", "color": "nope",
             "altitude": 9999},
        ]}, 50)
        assert len(out["corrections"]) >= 3


# ==========================================================================
# API
# ==========================================================================


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(server, "SHOW_DIR", tmp / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp / "logs")
    monkeypatch.setattr(server, "_fleet", None)
    (tmp / "shows").mkdir()
    return TestClient(server.app)


def say(client, text, show=None, confirmed=False, selected=0):
    return client.post("/api/voice", json={
        "text": text,
        "context": {"show": show, "selected": selected},
        "confirmed": confirmed,
    }).json()


class TestVoiceApi:
    def test_vocabulary_endpoint(self, client):
        v = client.get("/api/voice/vocabulary").json()
        assert v["examples"] and v["shapes"] and v["effects"]
        assert set(v["refused"]) == {"arm", "start", "disarm"}

    def test_a_whole_show_can_be_built_by_voice(self, client):
        r = say(client, "create a show called voice demo with sixty drones")
        assert r["executed"]
        show = r["open_show"]

        for phrase in ("add a red heart at 55 metres", "add a blue sphere",
                       "apply a chase effect in cyan", "set the tempo to 120"):
            out = say(client, phrase, show)
            assert out["executed"], f"{phrase!r}: {out}"

        prepared = say(client, "prepare", show)
        assert prepared["executed"]
        show = prepared["open_show"]

        validated = say(client, "validate", show)
        assert validated["result"]["passed"], validated["speech"]
        assert "passed" in validated["speech"].lower()

    def test_confirm_intents_do_not_execute_unconfirmed(self, client):
        r = say(client, "create a show called c with ten drones")
        show = r["open_show"]
        out = say(client, "upload the plans", show)
        assert out["needs_confirmation"] is True
        assert out["executed"] is False
        assert out["confirm_prompt"]

    def test_client_actions_are_relayed_not_executed(self, client):
        out = say(client, "top view")
        assert out["client_action"] == {"name": "camera", "view": "top"}
        assert out["executed"] is False

    def test_status_without_anything_loaded(self, client):
        assert "nothing loaded" in say(client, "status")["speech"].lower()

    def test_help_lists_the_refusal(self, client):
        out = say(client, "help")
        assert "not available by voice" in out["speech"]

    def test_commands_needing_a_show_say_so(self, client):
        assert "open a show" in say(client, "validate")["speech"].lower()

    def test_a_failing_command_reports_rather_than_500s(self, client):
        r = say(client, "create a show called tiny with four drones")
        show = r["open_show"]
        out = say(client, "validate", show)
        # Four drones and no formations: an error, surfaced as speech.
        assert out.get("error") or "failed" in out["speech"].lower() \
            or out["result"]["passed"] is False

    def test_unknown_phrase_returns_200_with_guidance(self, client):
        out = say(client, "purple monkey dishwasher")
        assert out["intent"]["name"] == "unknown"
        assert out["speech"]


class TestComposeApi:
    def test_draft_without_building(self, client):
        r = client.post("/api/voice/compose", json={
            "description": "a romantic wedding show", "drone_count": 60,
        }).json()
        assert r["created"] is None
        assert r["spec"]["formations"]

    def test_build_produces_a_real_show(self, client):
        r = client.post("/api/voice/compose", json={
            "description": "a festive show with a heart and a star",
            "drone_count": 60, "create": True, "name": "composed",
        }).json()
        assert r["created"] == "composed"
        assert r["formations"]

        detail = client.get("/api/design/shows/composed").json()
        assert detail["drone_count"] == 60
        assert len(detail["formations"]) == len(r["formations"])
        assert "not flyable" in r["note"].lower()

    def test_a_composed_show_goes_through_the_normal_pipeline(self, client):
        """Composed output is not special-cased anywhere downstream."""
        client.post("/api/voice/compose", json={
            "description": "a calm show", "drone_count": 40,
            "create": True, "name": "calm one",
        })
        prepared = client.post("/api/design/shows/calm one/prepare").json()
        report = client.post(
            f"/api/design/shows/{prepared['saved_as']}/validate").json()
        assert report["passed"], [v["message"] for v in report["violations"]]
        assert report["metrics"]["min_separation_m"] >= 3.0
