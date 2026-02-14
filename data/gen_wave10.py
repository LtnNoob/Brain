#!/usr/bin/env python3
"""Generate Wave 10: Astronomy + Physics Deep knowledge graph."""
import json

concepts = []
relations = []

def C(label, definition, typ="FACT", trust=0.95):
    concepts.append({"label": label, "definition": definition, "type": typ, "trust": round(trust, 2)})

def R(src, tgt, rel="RELATES_TO", w=0.9):
    relations.append({"source": src, "target": tgt, "type": rel, "weight": round(w, 2)})

# ============================================================
# SOLAR SYSTEM - PLANETS
# ============================================================
planets = [
    ("Mercury", "Smallest planet, closest to Sun, no atmosphere, extreme temperatures", 0.97),
    ("Venus", "Second planet from Sun, hottest planet due to greenhouse effect, retrograde rotation", 0.97),
    ("Earth", "Third planet from Sun, only known planet with life, liquid water on surface", 0.97),
    ("Mars", "Fourth planet, red planet, thin CO2 atmosphere, largest volcano Olympus Mons", 0.97),
    ("Jupiter", "Largest planet, gas giant, Great Red Spot storm, 95 known moons", 0.97),
    ("Saturn", "Sixth planet, gas giant with prominent ring system, least dense planet", 0.97),
    ("Uranus", "Seventh planet, ice giant, extreme axial tilt of 98 degrees, retrograde rotation", 0.97),
    ("Neptune", "Eighth planet, ice giant, strongest winds in solar system up to 2100 km/h", 0.97),
]
for name, defn, t in planets:
    C(name, defn, "FACT", t)
    R(name, "Solar System", "PART_OF", 0.99)
    R(name, "Planet", "IS_A", 0.99)

C("Planet", "Celestial body orbiting a star, massive enough for gravity to make it spherical, cleared its orbit", "DEFINITION", 0.99)
C("Dwarf Planet", "Celestial body orbiting Sun, spherical but has not cleared its orbital neighborhood", "DEFINITION", 0.98)
C("Solar System", "Gravitationally bound system of the Sun and objects orbiting it", "DEFINITION", 0.99)
C("Sun", "G-type main-sequence star at center of Solar System, 4.6 billion years old", "FACT", 0.99)
R("Sun", "Solar System", "PART_OF", 0.99)
R("Sun", "G-type Main-Sequence Star", "IS_A", 0.97)

dwarf_planets = ["Pluto", "Eris", "Haumea", "Makemake", "Ceres"]
dwarf_defs = [
    "Dwarf planet in Kuiper Belt, 5 moons, formerly classified as ninth planet",
    "Most massive dwarf planet, slightly smaller than Pluto, in scattered disc",
    "Egg-shaped dwarf planet in Kuiper Belt with two moons and a ring",
    "Second-largest Kuiper Belt object, reddish surface, one known moon",
    "Largest object in asteroid belt, only dwarf planet in inner solar system",
]
for name, defn in zip(dwarf_planets, dwarf_defs):
    C(name, defn, "FACT", 0.96)
    R(name, "Dwarf Planet", "IS_A", 0.98)
    R(name, "Solar System", "PART_OF", 0.98)

# ============================================================
# MOONS (50+)
# ============================================================
moons = [
    # Earth
    ("Moon", "Earth's only natural satellite, fifth largest moon in Solar System, tidally locked", "Earth"),
    # Mars
    ("Phobos", "Larger moon of Mars, irregularly shaped, orbiting closer than any other known moon to its planet", "Mars"),
    ("Deimos", "Smaller moon of Mars, irregularly shaped, smooth surface", "Mars"),
    # Jupiter moons (Galilean + others)
    ("Io", "Innermost Galilean moon of Jupiter, most volcanically active body in Solar System", "Jupiter"),
    ("Europa", "Galilean moon of Jupiter, ice-covered surface with subsurface ocean, potential for life", "Jupiter"),
    ("Ganymede", "Largest moon in Solar System, Galilean moon of Jupiter, has magnetic field", "Jupiter"),
    ("Callisto", "Outermost Galilean moon of Jupiter, most heavily cratered object in Solar System", "Jupiter"),
    ("Amalthea", "Fifth moon of Jupiter, irregularly shaped, reddish color", "Jupiter"),
    ("Himalia", "Largest irregular satellite of Jupiter, prograde orbit", "Jupiter"),
    ("Thebe", "Inner moon of Jupiter, irregularly shaped", "Jupiter"),
    ("Metis", "Innermost known moon of Jupiter, within main ring", "Jupiter"),
    ("Elara", "Prograde irregular satellite of Jupiter", "Jupiter"),
    ("Pasiphae", "Retrograde irregular satellite of Jupiter", "Jupiter"),
    ("Carme", "Retrograde irregular satellite of Jupiter, likely captured asteroid", "Jupiter"),
    ("Sinope", "Retrograde irregular satellite of Jupiter", "Jupiter"),
    ("Leda", "Small prograde satellite of Jupiter in Himalia group", "Jupiter"),
    # Saturn moons
    ("Titan", "Largest moon of Saturn, second largest in Solar System, dense nitrogen atmosphere with methane cycle", "Saturn"),
    ("Enceladus", "Moon of Saturn with water ice geysers, subsurface ocean, potential habitability", "Saturn"),
    ("Mimas", "Innermost major moon of Saturn, large Herschel crater resembles Death Star", "Saturn"),
    ("Tethys", "Mid-sized moon of Saturn with large Odysseus crater and Ithaca Chasma", "Saturn"),
    ("Dione", "Moon of Saturn with ice cliffs and wispy terrain", "Saturn"),
    ("Rhea", "Second-largest moon of Saturn, icy surface, possible tenuous ring system", "Saturn"),
    ("Iapetus", "Moon of Saturn with extreme two-tone coloration and equatorial ridge", "Saturn"),
    ("Hyperion", "Irregularly shaped moon of Saturn with sponge-like appearance, chaotic rotation", "Saturn"),
    ("Phoebe", "Retrograde irregular moon of Saturn, likely captured Kuiper Belt object", "Saturn"),
    ("Janus", "Co-orbital moon of Saturn sharing orbit with Epimetheus", "Saturn"),
    ("Epimetheus", "Co-orbital moon of Saturn sharing orbit with Janus", "Saturn"),
    ("Prometheus", "Inner shepherd moon of Saturn F ring", "Saturn"),
    ("Pandora", "Outer shepherd moon of Saturn F ring", "Saturn"),
    # Uranus moons
    ("Miranda", "Innermost major moon of Uranus, extreme geological features including Verona Rupes cliff", "Uranus"),
    ("Ariel", "Brightest moon of Uranus, extensive valley systems", "Uranus"),
    ("Umbriel", "Darkest major moon of Uranus, heavily cratered", "Uranus"),
    ("Titania", "Largest moon of Uranus, enormous canyon systems", "Uranus"),
    ("Oberon", "Outermost major moon of Uranus, heavily cratered with dark floor craters", "Uranus"),
    ("Puck", "Inner moon of Uranus discovered by Voyager 2", "Uranus"),
    ("Cordelia", "Innermost known moon of Uranus, inner shepherd of epsilon ring", "Uranus"),
    ("Ophelia", "Moon of Uranus, outer shepherd of epsilon ring", "Uranus"),
    # Neptune moons
    ("Triton", "Largest moon of Neptune, retrograde orbit suggests capture, nitrogen geysers, very cold surface", "Neptune"),
    ("Proteus", "Second-largest moon of Neptune, irregularly shaped", "Neptune"),
    ("Nereid", "Third-largest moon of Neptune, highly eccentric orbit", "Neptune"),
    ("Naiad", "Innermost moon of Neptune", "Neptune"),
    ("Thalassa", "Small inner moon of Neptune", "Neptune"),
    ("Despina", "Small inner moon of Neptune", "Neptune"),
    ("Galatea", "Inner moon of Neptune, shepherd moon of Adams ring", "Neptune"),
    ("Larissa", "Moon of Neptune, irregularly shaped", "Neptune"),
    # Pluto moons
    ("Charon", "Largest moon of Pluto, half its size, tidally locked binary system", "Pluto"),
    ("Nix", "Small moon of Pluto discovered in 2005", "Pluto"),
    ("Hydra", "Small moon of Pluto discovered in 2005", "Pluto"),
    ("Kerberos", "Small moon of Pluto discovered in 2011", "Pluto"),
    ("Styx", "Smallest moon of Pluto discovered in 2012", "Pluto"),
]
for name, defn, parent in moons:
    C(name, defn, "FACT", 0.95)
    R(name, "Natural Satellite", "IS_A", 0.97)
    R(name, parent, "PART_OF", 0.98)

C("Natural Satellite", "Celestial body orbiting a planet or dwarf planet, held by gravity", "DEFINITION", 0.98)

# ============================================================
# ALL 88 CONSTELLATIONS
# ============================================================
constellations = [
    ("Andromeda", "Princess chained, contains Andromeda Galaxy M31"),
    ("Antlia", "Air pump, faint southern constellation"),
    ("Apus", "Bird of paradise, near south celestial pole"),
    ("Aquarius", "Water bearer, zodiac constellation, contains Helix Nebula"),
    ("Aquila", "Eagle, contains bright star Altair"),
    ("Ara", "Altar, southern constellation in Milky Way"),
    ("Aries", "Ram, zodiac constellation"),
    ("Auriga", "Charioteer, contains bright star Capella"),
    ("Boötes", "Herdsman, contains bright star Arcturus"),
    ("Caelum", "Chisel, faint southern constellation"),
    ("Camelopardalis", "Giraffe, large faint northern constellation"),
    ("Cancer", "Crab, zodiac constellation, contains Beehive Cluster"),
    ("Canes Venatici", "Hunting dogs, contains Whirlpool Galaxy M51"),
    ("Canis Major", "Greater dog, contains Sirius brightest star in sky"),
    ("Canis Minor", "Lesser dog, contains Procyon"),
    ("Capricornus", "Sea goat, zodiac constellation"),
    ("Carina", "Keel of ship, contains Canopus second brightest star"),
    ("Cassiopeia", "Queen, W-shaped asterism in northern sky"),
    ("Centaurus", "Centaur, contains Alpha Centauri nearest star system"),
    ("Cepheus", "King, contains prototype Cepheid variable star"),
    ("Cetus", "Sea monster, contains Mira variable star"),
    ("Chamaeleon", "Chameleon, small southern constellation"),
    ("Circinus", "Compass, small southern constellation"),
    ("Columba", "Dove, southern constellation"),
    ("Coma Berenices", "Berenices hair, contains many galaxies in Coma Cluster"),
    ("Corona Australis", "Southern crown"),
    ("Corona Borealis", "Northern crown"),
    ("Corvus", "Crow, small southern constellation"),
    ("Crater", "Cup, faint constellation"),
    ("Crux", "Southern cross, smallest constellation, used for navigation"),
    ("Cygnus", "Swan, contains Deneb, Northern Cross asterism"),
    ("Delphinus", "Dolphin, small northern constellation"),
    ("Dorado", "Swordfish, contains Large Magellanic Cloud"),
    ("Draco", "Dragon, contains north ecliptic pole"),
    ("Equuleus", "Little horse, second smallest constellation"),
    ("Eridanus", "River, long constellation from Orion to south"),
    ("Fornax", "Furnace, contains Fornax Galaxy Cluster"),
    ("Gemini", "Twins, zodiac constellation, contains Castor and Pollux"),
    ("Grus", "Crane, southern constellation"),
    ("Hercules", "Hero, contains great globular cluster M13"),
    ("Horologium", "Clock, faint southern constellation"),
    ("Hydra", "Water snake, largest constellation by area"),
    ("Hydrus", "Male water snake, southern constellation"),
    ("Indus", "Indian, southern constellation"),
    ("Lacerta", "Lizard, northern constellation"),
    ("Leo", "Lion, zodiac constellation, contains Regulus"),
    ("Leo Minor", "Smaller lion, faint northern constellation"),
    ("Lepus", "Hare, below Orion"),
    ("Libra", "Scales, zodiac constellation"),
    ("Lupus", "Wolf, southern constellation"),
    ("Lynx", "Lynx, faint northern constellation"),
    ("Lyra", "Lyre, contains Vega and Ring Nebula M57"),
    ("Mensa", "Table Mountain, faintest constellation, contains part of LMC"),
    ("Microscopium", "Microscope, faint southern constellation"),
    ("Monoceros", "Unicorn, on celestial equator in Milky Way"),
    ("Musca", "Fly, small southern constellation"),
    ("Norma", "Carpenters square, southern Milky Way"),
    ("Octans", "Octant, contains south celestial pole"),
    ("Ophiuchus", "Serpent bearer, large equatorial constellation"),
    ("Orion", "Hunter, prominent constellation with Betelgeuse and Rigel, Orion Nebula"),
    ("Pavo", "Peacock, southern constellation"),
    ("Pegasus", "Winged horse, Great Square asterism"),
    ("Perseus", "Hero, contains Algol eclipsing binary and Double Cluster"),
    ("Phoenix", "Phoenix, southern constellation"),
    ("Pictor", "Painters easel, contains Beta Pictoris with debris disk"),
    ("Pisces", "Fishes, zodiac constellation"),
    ("Piscis Austrinus", "Southern fish, contains Fomalhaut"),
    ("Puppis", "Stern of ship, in Milky Way"),
    ("Pyxis", "Compass box, faint constellation"),
    ("Reticulum", "Net, small southern constellation"),
    ("Sagitta", "Arrow, third smallest constellation"),
    ("Sagittarius", "Archer, zodiac constellation, direction of galactic center"),
    ("Scorpius", "Scorpion, zodiac constellation, contains Antares"),
    ("Sculptor", "Sculptor, contains south galactic pole"),
    ("Scutum", "Shield, small constellation in Milky Way"),
    ("Serpens", "Serpent, only constellation in two parts Caput and Cauda"),
    ("Sextans", "Sextant, faint equatorial constellation"),
    ("Taurus", "Bull, zodiac constellation, contains Pleiades and Hyades, Crab Nebula"),
    ("Telescopium", "Telescope, faint southern constellation"),
    ("Triangulum", "Triangle, contains Triangulum Galaxy M33"),
    ("Triangulum Australe", "Southern triangle"),
    ("Tucana", "Toucan, contains Small Magellanic Cloud and 47 Tucanae"),
    ("Ursa Major", "Great bear, contains Big Dipper asterism"),
    ("Ursa Minor", "Little bear, contains Polaris north star"),
    ("Vela", "Sails of ship, contains Vela pulsar"),
    ("Virgo", "Maiden, zodiac constellation, contains Virgo Galaxy Cluster"),
    ("Volans", "Flying fish, southern constellation"),
    ("Vulpecula", "Fox, contains Dumbbell Nebula M27"),
]
for name, defn in constellations:
    C(name, f"Constellation: {defn}", "FACT", 0.96)
    R(name, "Constellation", "IS_A", 0.99)

C("Constellation", "Defined area of celestial sphere, 88 officially recognized by IAU", "DEFINITION", 0.99)
C("Zodiac", "Band of 12 constellations along ecliptic through which Sun Moon and planets appear to move", "DEFINITION", 0.97)
zodiac = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpius","Sagittarius","Capricornus","Aquarius","Pisces"]
for z in zodiac:
    R(z, "Zodiac", "PART_OF", 0.95)

# ============================================================
# STAR TYPES
# ============================================================
star_types = [
    ("O-type Star", "Hottest main-sequence star, blue, >30000K, very massive and luminous, rare", "DEFINITION"),
    ("B-type Star", "Hot blue-white star, 10000-30000K, includes many bright stars like Rigel", "DEFINITION"),
    ("A-type Star", "White star, 7500-10000K, includes Sirius and Vega", "DEFINITION"),
    ("F-type Star", "Yellow-white star, 6000-7500K, includes Procyon", "DEFINITION"),
    ("G-type Main-Sequence Star", "Yellow star like Sun, 5200-6000K, main-sequence hydrogen fusion", "DEFINITION"),
    ("K-type Star", "Orange star, 3700-5200K, includes Alpha Centauri B", "DEFINITION"),
    ("M-type Star", "Red dwarf, coolest main-sequence, <3700K, most common star type", "DEFINITION"),
    ("Red Giant", "Evolved star that exhausted core hydrogen, expanded and cooled, luminosity class III", "DEFINITION"),
    ("Red Supergiant", "Massive evolved star, enormous radius, includes Betelgeuse", "DEFINITION"),
    ("Blue Supergiant", "Massive hot luminous star, short-lived, includes Rigel", "DEFINITION"),
    ("White Dwarf", "Stellar remnant of low-mass star, electron degeneracy pressure, Earth-sized", "DEFINITION"),
    ("Neutron Star", "Extremely dense stellar remnant, 1.4-2 solar masses in 10km radius, from supernova", "DEFINITION"),
    ("Pulsar", "Rotating neutron star emitting beams of electromagnetic radiation", "DEFINITION"),
    ("Magnetar", "Neutron star with extremely powerful magnetic field up to 10^15 gauss", "DEFINITION"),
    ("Black Hole", "Region of spacetime where gravity prevents anything including light from escaping", "DEFINITION"),
    ("Stellar Black Hole", "Black hole formed from gravitational collapse of massive star, 3-100 solar masses", "DEFINITION"),
    ("Supermassive Black Hole", "Black hole with mass millions to billions of solar masses at galaxy centers", "DEFINITION"),
    ("Brown Dwarf", "Substellar object too low mass for hydrogen fusion, 13-80 Jupiter masses", "DEFINITION"),
    ("T Tauri Star", "Young pre-main-sequence star, variable, still contracting", "DEFINITION"),
    ("Wolf-Rayet Star", "Massive evolved star with strong stellar wind, losing outer hydrogen", "DEFINITION"),
    ("Cepheid Variable", "Pulsating variable star with period-luminosity relation used as distance indicator", "DEFINITION"),
    ("Binary Star", "Star system of two stars orbiting common center of mass", "DEFINITION"),
    ("Eclipsing Binary", "Binary star system where components eclipse each other from observers perspective", "DEFINITION"),
    ("Nova", "Thermonuclear explosion on white dwarf surface accreting matter from companion", "DEFINITION"),
    ("Supernova", "Catastrophic stellar explosion, either core collapse or thermonuclear, extreme luminosity", "DEFINITION"),
    ("Type Ia Supernova", "Thermonuclear supernova from white dwarf exceeding Chandrasekhar limit, standard candle", "DEFINITION"),
    ("Type II Supernova", "Core-collapse supernova of massive star, hydrogen lines in spectrum", "DEFINITION"),
    ("Hypernova", "Extremely energetic supernova, energy >10^46 joules, associated with gamma-ray bursts", "DEFINITION"),
    ("Protostar", "Early stage of star formation, gravitationally collapsing cloud before nuclear fusion", "DEFINITION"),
    ("Main Sequence", "Band on HR diagram where stars spend most of life fusing hydrogen to helium", "DEFINITION"),
    ("Hertzsprung-Russell Diagram", "Scatter plot of stellar luminosity vs temperature showing stellar evolution stages", "DEFINITION"),
]
for name, defn, typ in star_types:
    C(name, defn, typ, 0.96)
    R(name, "Star", "IS_A" if "Star" in name or name in ["Pulsar","Magnetar","Protostar"] else "RELATES_TO", 0.9)

C("Star", "Luminous sphere of plasma held together by gravity undergoing nuclear fusion", "DEFINITION", 0.99)
R("Protostar", "Main Sequence", "ENABLES", 0.9)
R("Main Sequence", "Red Giant", "ENABLES", 0.9)
R("Red Giant", "White Dwarf", "ENABLES", 0.88)
R("Red Supergiant", "Supernova", "CAUSES", 0.92)
R("Supernova", "Neutron Star", "CAUSES", 0.9)
R("Supernova", "Stellar Black Hole", "CAUSES", 0.88)
R("Neutron Star", "Pulsar", "IS_A", 0.85)
R("Neutron Star", "Magnetar", "IS_A", 0.85)
R("Cepheid Variable", "Cosmic Distance Measurement", "USED_IN", 0.93)
R("Type Ia Supernova", "Cosmic Distance Measurement", "USED_IN", 0.93)

# ============================================================
# GALAXY TYPES
# ============================================================
galaxy_types = [
    ("Galaxy", "Gravitationally bound system of stars, gas, dust, and dark matter", "DEFINITION"),
    ("Spiral Galaxy", "Galaxy with spiral arms winding from central bulge, includes Milky Way", "DEFINITION"),
    ("Barred Spiral Galaxy", "Spiral galaxy with central bar-shaped structure, includes Milky Way", "DEFINITION"),
    ("Elliptical Galaxy", "Galaxy with ellipsoidal shape, little gas/dust, old stars, range from dwarf to giant", "DEFINITION"),
    ("Lenticular Galaxy", "Intermediate galaxy between elliptical and spiral, disk but no arms", "DEFINITION"),
    ("Irregular Galaxy", "Galaxy without distinct regular shape, includes Magellanic Clouds", "DEFINITION"),
    ("Dwarf Galaxy", "Small galaxy with few billion stars, most common galaxy type", "DEFINITION"),
    ("Active Galaxy", "Galaxy with unusually high luminosity from active galactic nucleus", "DEFINITION"),
    ("Seyfert Galaxy", "Spiral galaxy with bright active nucleus emitting broad emission lines", "DEFINITION"),
    ("Quasar", "Extremely luminous active galactic nucleus powered by supermassive black hole accretion", "DEFINITION"),
    ("Blazar", "Active galaxy with relativistic jet pointed toward Earth", "DEFINITION"),
    ("Radio Galaxy", "Galaxy emitting large amounts of radio waves from relativistic jets", "DEFINITION"),
    ("Starburst Galaxy", "Galaxy undergoing exceptionally high rate of star formation", "DEFINITION"),
    ("Milky Way", "Barred spiral galaxy containing Solar System, 100-400 billion stars, 100000 ly diameter", "FACT"),
    ("Andromeda Galaxy", "Nearest large spiral galaxy M31, 2.5 million ly away, approaching Milky Way", "FACT"),
    ("Large Magellanic Cloud", "Irregular dwarf galaxy satellite of Milky Way, visible from southern hemisphere", "FACT"),
    ("Small Magellanic Cloud", "Dwarf galaxy satellite of Milky Way near Large Magellanic Cloud", "FACT"),
    ("Hubble Classification", "Galaxy morphological classification scheme: elliptical, spiral, barred spiral, irregular", "DEFINITION"),
]
for name, defn, typ in galaxy_types:
    C(name, defn, typ, 0.96)

R("Spiral Galaxy", "Galaxy", "IS_A", 0.99)
R("Barred Spiral Galaxy", "Spiral Galaxy", "IS_A", 0.98)
R("Elliptical Galaxy", "Galaxy", "IS_A", 0.99)
R("Lenticular Galaxy", "Galaxy", "IS_A", 0.99)
R("Irregular Galaxy", "Galaxy", "IS_A", 0.99)
R("Dwarf Galaxy", "Galaxy", "IS_A", 0.98)
R("Milky Way", "Barred Spiral Galaxy", "IS_A", 0.97)
R("Andromeda Galaxy", "Spiral Galaxy", "IS_A", 0.96)
R("Solar System", "Milky Way", "PART_OF", 0.99)
R("Supermassive Black Hole", "Active Galaxy", "ENABLES", 0.93)
R("Quasar", "Active Galaxy", "IS_A", 0.9)
R("Seyfert Galaxy", "Active Galaxy", "IS_A", 0.9)
R("Blazar", "Active Galaxy", "IS_A", 0.9)

# ============================================================
# SPACE MISSIONS (100+)
# ============================================================
missions = [
    # Early Space Age
    ("Sputnik 1", "First artificial satellite, launched by USSR 1957, started Space Age", "FACT"),
    ("Sputnik 2", "Second satellite, carried dog Laika first animal in orbit 1957", "FACT"),
    ("Explorer 1", "First US satellite 1958, discovered Van Allen radiation belts", "FACT"),
    ("Vanguard 1", "Fourth satellite in orbit 1958, oldest still in orbit", "FACT"),
    ("Luna 1", "First spacecraft to reach vicinity of Moon 1959 USSR", "FACT"),
    ("Luna 2", "First spacecraft to impact Moon 1959 USSR", "FACT"),
    ("Luna 3", "First spacecraft to photograph far side of Moon 1959", "FACT"),
    ("Vostok 1", "First human spaceflight, Yuri Gagarin 1961", "FACT"),
    ("Mercury-Redstone 3", "First American in space, Alan Shepard 1961 suborbital", "FACT"),
    ("Mercury-Atlas 6", "John Glenn first American to orbit Earth 1962", "FACT"),
    ("Vostok 6", "Valentina Tereshkova first woman in space 1963", "FACT"),
    ("Voskhod 2", "First spacewalk by Alexei Leonov 1965", "FACT"),
    ("Mariner 2", "First successful planetary flyby Venus 1962", "FACT"),
    ("Mariner 4", "First Mars flyby with close-up images 1965", "FACT"),
    ("Mariner 9", "First spacecraft to orbit another planet Mars 1971", "FACT"),
    ("Mariner 10", "First Mercury flyby 1974, first gravity assist", "FACT"),
    ("Gemini 4", "First American spacewalk Ed White 1965", "FACT"),
    ("Gemini 8", "First orbital docking Neil Armstrong 1966", "FACT"),
    # Apollo
    ("Apollo 7", "First crewed Apollo mission Earth orbit 1968", "FACT"),
    ("Apollo 8", "First crewed spacecraft to orbit Moon 1968", "FACT"),
    ("Apollo 11", "First Moon landing, Neil Armstrong and Buzz Aldrin July 20 1969", "FACT"),
    ("Apollo 12", "Second Moon landing November 1969 precision landing near Surveyor 3", "FACT"),
    ("Apollo 13", "Failed Moon landing 1970, successful emergency return, successful failure", "FACT"),
    ("Apollo 14", "Third Moon landing 1971 Alan Shepard", "FACT"),
    ("Apollo 15", "Fourth Moon landing 1971 first lunar rover", "FACT"),
    ("Apollo 16", "Fifth Moon landing 1972 lunar highlands", "FACT"),
    ("Apollo 17", "Last Moon landing December 1972, longest lunar surface stay", "FACT"),
    # Space stations
    ("Salyut 1", "First space station launched 1971 USSR", "FACT"),
    ("Skylab", "First American space station 1973-1979", "FACT"),
    ("Mir", "Soviet/Russian space station 1986-2001, long-duration missions", "FACT"),
    ("International Space Station", "Largest structure in space, multinational collaboration since 1998, low Earth orbit", "FACT"),
    ("Tiangong", "Chinese space station, modular construction began 2021", "FACT"),
    # Shuttle era
    ("STS-1", "First Space Shuttle flight Columbia April 1981", "FACT"),
    ("STS-31", "Space Shuttle Discovery deployed Hubble Space Telescope 1990", "FACT"),
    ("STS-61", "First Hubble servicing mission 1993, corrected optics", "FACT"),
    ("STS-107", "Columbia disaster during reentry February 2003, 7 crew lost", "FACT"),
    ("STS-51-L", "Challenger disaster 73 seconds after launch January 1986, 7 crew lost", "FACT"),
    ("STS-135", "Final Space Shuttle mission Atlantis July 2011", "FACT"),
    # Planetary probes
    ("Viking 1", "First successful Mars lander 1976, searched for life", "FACT"),
    ("Viking 2", "Second Mars lander 1976, surface operations at Utopia Planitia", "FACT"),
    ("Voyager 1", "Launched 1977, farthest human-made object, visited Jupiter Saturn, now interstellar space", "FACT"),
    ("Voyager 2", "Launched 1977, only spacecraft to visit all four giant planets", "FACT"),
    ("Pioneer 10", "First spacecraft to Jupiter 1973, first beyond asteroid belt", "FACT"),
    ("Pioneer 11", "First Saturn flyby 1979", "FACT"),
    ("Venera 7", "First successful landing on another planet Venus 1970", "FACT"),
    ("Venera 9", "First images from surface of another planet Venus 1975", "FACT"),
    ("Venera 13", "First color images and sound recording from Venus surface 1982", "FACT"),
    ("Magellan", "Mapped 98% of Venus surface with radar 1990-1994", "FACT"),
    ("Galileo", "First Jupiter orbiter 1995-2003, discovered subsurface oceans on Europa", "FACT"),
    ("Cassini-Huygens", "Saturn orbiter 2004-2017, Huygens probe landed on Titan", "FACT"),
    ("Huygens", "ESA probe landed on Titan 2005, first landing in outer solar system", "FACT"),
    ("Mars Pathfinder", "First successful Mars rover Sojourner 1997", "FACT"),
    ("Mars Exploration Rover Spirit", "Mars rover 2004-2010, Gusev Crater", "FACT"),
    ("Mars Exploration Rover Opportunity", "Mars rover 2004-2018, marathon distance, found evidence of water", "FACT"),
    ("Mars Science Laboratory Curiosity", "Large Mars rover since 2012, Gale Crater, found organic molecules", "FACT"),
    ("Mars 2020 Perseverance", "Mars rover since 2021, Jezero Crater, sample collection for return", "FACT"),
    ("Ingenuity", "First powered flight on another planet, Mars helicopter 2021", "FACT"),
    ("Phoenix", "Mars lander 2008, confirmed water ice in arctic soil", "FACT"),
    ("InSight", "Mars lander 2018-2022, seismometer measured marsquakes", "FACT"),
    ("MAVEN", "Mars atmosphere and volatile evolution orbiter since 2014", "FACT"),
    ("Mars Reconnaissance Orbiter", "High-resolution Mars orbiter since 2006", "FACT"),
    ("Mars Express", "ESA Mars orbiter since 2003", "FACT"),
    ("New Horizons", "First Pluto flyby July 2015, then Kuiper Belt object Arrokoth 2019", "FACT"),
    ("Juno", "Jupiter polar orbiter since 2016, studying interior and magnetosphere", "FACT"),
    ("Dawn", "Orbited Vesta and Ceres, first to orbit two extraterrestrial bodies", "FACT"),
    ("MESSENGER", "First Mercury orbiter 2011-2015, mapped entire surface", "FACT"),
    ("BepiColombo", "ESA-JAXA Mercury mission launched 2018, arrival 2025", "FACT"),
    ("Rosetta", "ESA mission, first to orbit comet 67P and land Philae probe 2014", "FACT"),
    ("Philae", "First spacecraft to land on comet 67P/Churyumov-Gerasimenko 2014", "FACT"),
    ("Hayabusa", "JAXA asteroid sample return from Itokawa 2010", "FACT"),
    ("Hayabusa2", "JAXA sample return from asteroid Ryugu 2020", "FACT"),
    ("OSIRIS-REx", "NASA asteroid sample return from Bennu 2023", "FACT"),
    ("Stardust", "Collected comet Wild 2 dust samples returned 2006", "FACT"),
    ("Deep Impact", "Impacted comet Tempel 1 to study interior composition 2005", "FACT"),
    ("NEAR Shoemaker", "First to orbit and land on asteroid Eros 2001", "FACT"),
    ("Genesis", "Collected solar wind samples returned to Earth 2004", "FACT"),
    ("Parker Solar Probe", "Closest approach to Sun, studying solar corona and wind since 2018", "FACT"),
    ("Solar Orbiter", "ESA solar observation mission launched 2020", "FACT"),
    ("SOHO", "Solar and Heliospheric Observatory at L1 since 1996", "FACT"),
    ("Ulysses", "First spacecraft to fly over Sun poles 1990-2009", "FACT"),
    # Telescopes
    ("Hubble Space Telescope", "Optical space telescope launched 1990, revolutionized astronomy", "FACT"),
    ("James Webb Space Telescope", "Infrared space telescope launched 2021 at L2, successor to Hubble", "FACT"),
    ("Chandra X-ray Observatory", "NASA X-ray space telescope launched 1999", "FACT"),
    ("Spitzer Space Telescope", "Infrared space telescope 2003-2020", "FACT"),
    ("Kepler Space Telescope", "Discovered thousands of exoplanets using transit method 2009-2018", "FACT"),
    ("TESS", "Transiting Exoplanet Survey Satellite, all-sky exoplanet survey since 2018", "FACT"),
    ("Fermi Gamma-ray Space Telescope", "Gamma-ray observatory since 2008", "FACT"),
    ("Planck", "ESA cosmic microwave background surveyor 2009-2013", "FACT"),
    ("WMAP", "Wilkinson Microwave Anisotropy Probe, CMB measurements 2001-2010", "FACT"),
    ("COBE", "Cosmic Background Explorer, first CMB anisotropy detection 1989-1993", "FACT"),
    ("Gaia", "ESA astrometry mission mapping billion stars since 2013", "FACT"),
    ("WISE", "Wide-field Infrared Survey Explorer, all-sky infrared survey 2009", "FACT"),
    # Recent/future
    ("Europa Clipper", "NASA mission to study Jupiter moon Europa habitability launched 2024", "FACT"),
    ("JUICE", "ESA Jupiter Icy Moons Explorer launched 2023", "FACT"),
    ("Lucy", "NASA mission to study Jupiter Trojan asteroids launched 2021", "FACT"),
    ("DART", "Double Asteroid Redirection Test, first planetary defense mission, hit Dimorphos 2022", "FACT"),
    ("Artemis I", "Uncrewed test flight of SLS and Orion around Moon 2022", "FACT"),
    ("Artemis II", "First crewed Artemis mission, lunar flyby planned 2025", "FACT"),
    ("Chang'e 4", "First landing on far side of Moon 2019 China", "FACT"),
    ("Chang'e 5", "Lunar sample return mission 2020 China", "FACT"),
    ("Chandrayaan-3", "India successful lunar south pole landing 2023", "FACT"),
    ("SLIM", "JAXA Smart Lander for Investigating Moon, precision landing 2024", "FACT"),
    ("Dragonfly", "NASA rotorcraft mission to Titan planned launch 2028", "FACT"),
    ("Psyche", "NASA mission to metallic asteroid 16 Psyche launched 2023", "FACT"),
    ("VERITAS", "NASA Venus orbiter radar mapper planned", "FACT"),
    ("DAVINCI", "NASA Venus atmosphere probe planned", "FACT"),
    ("Starship", "SpaceX fully reusable super heavy-lift launch vehicle, largest ever", "FACT"),
    ("Crew Dragon", "SpaceX crewed spacecraft for ISS missions since 2020", "FACT"),
    ("Starliner", "Boeing crewed spacecraft for ISS missions", "FACT"),
]
for name, defn, typ in missions:
    C(name, defn, typ, 0.95)
    R(name, "Space Mission", "IS_A", 0.95)

C("Space Mission", "Organized endeavor to explore space using spacecraft", "DEFINITION", 0.98)
# Mission target relations
mission_targets = {
    "Moon": ["Apollo 11","Apollo 12","Apollo 13","Apollo 14","Apollo 15","Apollo 16","Apollo 17","Apollo 8","Luna 1","Luna 2","Luna 3","Chang'e 4","Chang'e 5","Chandrayaan-3","Artemis I","SLIM"],
    "Mars": ["Viking 1","Viking 2","Mars Pathfinder","Mars Exploration Rover Spirit","Mars Exploration Rover Opportunity","Mars Science Laboratory Curiosity","Mars 2020 Perseverance","Ingenuity","Phoenix","InSight","MAVEN","Mars Reconnaissance Orbiter","Mars Express","Mariner 4","Mariner 9"],
    "Venus": ["Venera 7","Venera 9","Venera 13","Magellan","Mariner 2"],
    "Jupiter": ["Galileo","Juno","Pioneer 10","Voyager 1","Voyager 2","Europa Clipper","JUICE","Lucy"],
    "Saturn": ["Cassini-Huygens","Pioneer 11","Voyager 1","Voyager 2"],
    "Mercury": ["Mariner 10","MESSENGER","BepiColombo"],
    "Pluto": ["New Horizons"],
    "Sun": ["Parker Solar Probe","Solar Orbiter","SOHO","Ulysses"],
}
for target, ms in mission_targets.items():
    for m in ms:
        R(m, target, "RELATES_TO", 0.92)

# ============================================================
# STANDARD MODEL PARTICLES
# ============================================================
# Quarks
quarks = [
    ("Up Quark", "Lightest quark, charge +2/3, mass ~2.2 MeV, constituent of protons and neutrons"),
    ("Down Quark", "Second lightest quark, charge -1/3, mass ~4.7 MeV, constituent of protons and neutrons"),
    ("Charm Quark", "Second generation quark, charge +2/3, mass ~1.27 GeV"),
    ("Strange Quark", "Second generation quark, charge -1/3, mass ~95 MeV"),
    ("Top Quark", "Heaviest quark and elementary particle, charge +2/3, mass ~173 GeV"),
    ("Bottom Quark", "Third generation quark, charge -1/3, mass ~4.18 GeV"),
]
for name, defn in quarks:
    C(name, defn, "FACT", 0.97)
    R(name, "Quark", "IS_A", 0.99)
    R(name, "Fermion", "IS_A", 0.99)

# Leptons
leptons = [
    ("Electron", "Lightest charged lepton, charge -1, mass 0.511 MeV, orbits atomic nucleus"),
    ("Muon", "Second generation charged lepton, charge -1, mass 105.7 MeV, unstable"),
    ("Tau", "Third generation charged lepton, charge -1, mass 1776.8 MeV, heaviest lepton"),
    ("Electron Neutrino", "Neutrino associated with electron, nearly massless, weak interaction only"),
    ("Muon Neutrino", "Neutrino associated with muon, nearly massless"),
    ("Tau Neutrino", "Neutrino associated with tau, nearly massless, last discovered 2000"),
]
for name, defn in leptons:
    C(name, defn, "FACT", 0.97)
    R(name, "Lepton", "IS_A", 0.99)
    R(name, "Fermion", "IS_A", 0.99)

# Gauge bosons
bosons = [
    ("Photon", "Gauge boson mediating electromagnetic force, massless, spin 1", "Electromagnetic Force"),
    ("W Boson", "Gauge boson mediating weak force, mass ~80.4 GeV, charged W+ and W-", "Weak Nuclear Force"),
    ("Z Boson", "Neutral gauge boson mediating weak force, mass ~91.2 GeV", "Weak Nuclear Force"),
    ("Gluon", "Gauge boson mediating strong force, massless, carries color charge, 8 types", "Strong Nuclear Force"),
    ("Higgs Boson", "Scalar boson giving mass to particles via Higgs mechanism, mass ~125 GeV, discovered 2012", None),
]
for name, defn, force in bosons:
    C(name, defn, "FACT", 0.97)
    R(name, "Boson", "IS_A", 0.99)
    if force:
        R(name, force, "ENABLES", 0.95)

# Composite particles
composites = [
    ("Proton", "Baryon with 2 up quarks and 1 down quark, charge +1, stable"),
    ("Neutron", "Baryon with 1 up quark and 2 down quarks, charge 0, half-life ~10 min free"),
    ("Pion", "Lightest meson, mediates residual strong force between nucleons"),
    ("Kaon", "Meson containing strange quark, important in CP violation studies"),
    ("Antiproton", "Antimatter counterpart of proton, charge -1"),
    ("Positron", "Antimatter counterpart of electron, charge +1, first antimatter discovered"),
]
for name, defn in composites:
    C(name, defn, "FACT", 0.97)

R("Proton", "Up Quark", "PART_OF", 0.99)
R("Proton", "Down Quark", "PART_OF", 0.99)
R("Neutron", "Up Quark", "PART_OF", 0.99)
R("Neutron", "Down Quark", "PART_OF", 0.99)
R("Proton", "Baryon", "IS_A", 0.99)
R("Neutron", "Baryon", "IS_A", 0.99)

particle_categories = [
    ("Quark", "Elementary particle and fundamental constituent of matter, six flavors, fractional charge", "DEFINITION"),
    ("Lepton", "Elementary particle not subject to strong force, includes electron and neutrinos", "DEFINITION"),
    ("Fermion", "Particle with half-integer spin obeying Fermi-Dirac statistics and Pauli exclusion", "DEFINITION"),
    ("Boson", "Particle with integer spin obeying Bose-Einstein statistics, force carriers", "DEFINITION"),
    ("Baryon", "Composite particle of three quarks, includes protons and neutrons", "DEFINITION"),
    ("Meson", "Composite particle of quark and antiquark", "DEFINITION"),
    ("Hadron", "Composite particle made of quarks bound by strong force", "DEFINITION"),
    ("Antimatter", "Matter composed of antiparticles with opposite charge and quantum numbers", "DEFINITION"),
    ("Standard Model", "Theory describing fundamental particles and three of four forces excluding gravity", "THEORY"),
]
for name, defn, typ in particle_categories:
    C(name, defn, typ, 0.97)

R("Baryon", "Hadron", "IS_A", 0.99)
R("Meson", "Hadron", "IS_A", 0.99)
R("Quark", "Fermion", "IS_A", 0.99)
R("Lepton", "Fermion", "IS_A", 0.99)
R("Hadron", "Quark", "PART_OF", 0.95)
R("Higgs Boson", "Higgs Mechanism", "ENABLES", 0.95)

# ============================================================
# FUNDAMENTAL FORCES
# ============================================================
forces = [
    ("Electromagnetic Force", "Force between charged particles, infinite range, mediated by photons, relative strength 1/137"),
    ("Strong Nuclear Force", "Strongest force, binds quarks in hadrons, mediated by gluons, range ~10^-15 m"),
    ("Weak Nuclear Force", "Force responsible for radioactive decay, mediated by W and Z bosons, very short range"),
    ("Gravitational Force", "Weakest fundamental force, infinite range, attractive between all mass-energy, described by general relativity"),
    ("Electroweak Force", "Unified electromagnetic and weak forces above ~100 GeV energy scale"),
]
for name, defn in forces:
    C(name, defn, "DEFINITION", 0.98)
    R(name, "Fundamental Force", "IS_A", 0.99)

C("Fundamental Force", "One of four fundamental interactions governing all physical phenomena", "DEFINITION", 0.99)
R("Electromagnetic Force", "Electroweak Force", "PART_OF", 0.9)
R("Weak Nuclear Force", "Electroweak Force", "PART_OF", 0.9)
R("Strong Nuclear Force", "Quark", "ENABLES", 0.95)
R("Gravitational Force", "General Relativity", "RELATES_TO", 0.95)

# ============================================================
# ENERGY TYPES
# ============================================================
energy_types = [
    ("Kinetic Energy", "Energy of motion, KE = 1/2 mv^2 for classical objects"),
    ("Potential Energy", "Stored energy due to position or configuration in a force field"),
    ("Gravitational Potential Energy", "Energy stored due to position in gravitational field, PE = mgh near surface"),
    ("Elastic Potential Energy", "Energy stored in deformed elastic material, PE = 1/2 kx^2"),
    ("Chemical Energy", "Energy stored in chemical bonds, released in reactions"),
    ("Nuclear Energy", "Energy stored in atomic nuclei, released in fission or fusion"),
    ("Thermal Energy", "Internal energy related to temperature, kinetic energy of particles"),
    ("Electrical Energy", "Energy from electric charge flow or electric fields"),
    ("Magnetic Energy", "Energy stored in magnetic field, E = B^2/2μ₀ per volume"),
    ("Electromagnetic Energy", "Energy carried by electromagnetic radiation"),
    ("Radiant Energy", "Energy of electromagnetic waves including light"),
    ("Sound Energy", "Energy carried by sound waves through mechanical vibrations"),
    ("Mechanical Energy", "Sum of kinetic and potential energy in a system"),
    ("Dark Energy", "Hypothetical energy causing accelerating expansion of universe, ~68% of universe"),
    ("Vacuum Energy", "Energy of empty space from quantum field fluctuations"),
    ("Rest Energy", "Energy equivalent of rest mass, E = mc^2"),
    ("Binding Energy", "Energy required to disassemble system into separate parts"),
    ("Ionization Energy", "Energy required to remove electron from atom or molecule"),
]
for name, defn in energy_types:
    C(name, defn, "DEFINITION", 0.96)
    R(name, "Energy", "IS_A", 0.97)

C("Energy", "Quantitative property transferred to or from a body to perform work or heat, conserved in isolated systems", "DEFINITION", 0.99)
R("Kinetic Energy", "Potential Energy", "RELATES_TO", 0.85)
R("Nuclear Energy", "Nuclear Fission", "ENABLES", 0.93)
R("Nuclear Energy", "Nuclear Fusion", "ENABLES", 0.93)

# ============================================================
# PHYSICS LAWS AND EQUATIONS (200+)
# ============================================================
physics_laws = [
    # Classical Mechanics
    ("Newton's First Law", "Object at rest stays at rest, object in motion stays in uniform motion unless acted on by net force", "THEORY"),
    ("Newton's Second Law", "Net force equals mass times acceleration F=ma, fundamental equation of classical mechanics", "THEORY"),
    ("Newton's Third Law", "Every action has equal and opposite reaction, forces always occur in pairs", "THEORY"),
    ("Newton's Law of Universal Gravitation", "Gravitational force between two masses F = GMm/r^2", "THEORY"),
    ("Kepler's First Law", "Planets orbit in ellipses with Sun at one focus", "THEORY"),
    ("Kepler's Second Law", "Line from Sun to planet sweeps equal areas in equal times", "THEORY"),
    ("Kepler's Third Law", "Square of orbital period proportional to cube of semi-major axis T^2 ∝ a^3", "THEORY"),
    ("Conservation of Energy", "Total energy in isolated system remains constant, energy cannot be created or destroyed", "THEORY"),
    ("Conservation of Momentum", "Total momentum of isolated system remains constant", "THEORY"),
    ("Conservation of Angular Momentum", "Total angular momentum remains constant when no external torque", "THEORY"),
    ("Conservation of Charge", "Total electric charge in isolated system remains constant", "THEORY"),
    ("Work-Energy Theorem", "Net work done on object equals change in kinetic energy W = ΔKE", "THEORY"),
    ("Impulse-Momentum Theorem", "Impulse equals change in momentum FΔt = Δp", "THEORY"),
    ("Hooke's Law", "Force of spring proportional to displacement F = -kx for small deformations", "THEORY"),
    ("Archimedes' Principle", "Buoyant force equals weight of displaced fluid", "THEORY"),
    ("Pascal's Law", "Pressure change applied to enclosed fluid transmitted equally throughout", "THEORY"),
    ("Bernoulli's Principle", "Increase in fluid speed occurs with decrease in pressure or potential energy", "THEORY"),
    ("Torricelli's Theorem", "Speed of fluid flowing from orifice v = sqrt(2gh)", "THEORY"),
    ("Continuity Equation", "Mass flow rate constant in steady flow A₁v₁ = A₂v₂ for incompressible fluid", "THEORY"),
    ("Navier-Stokes Equations", "Equations of motion for viscous fluid flow, millennium prize problem", "THEORY"),
    ("Euler's Equations of Motion", "Equations for inviscid fluid flow", "THEORY"),
    ("D'Alembert's Principle", "Sum of forces minus inertial forces equals zero for dynamic systems", "THEORY"),
    ("Hamilton's Principle", "Actual path minimizes action integral, foundation of Lagrangian mechanics", "THEORY"),
    ("Lagrangian Mechanics", "Reformulation of mechanics using generalized coordinates and Lagrangian L=T-V", "THEORY"),
    ("Hamiltonian Mechanics", "Reformulation of mechanics using Hamilton's equations with phase space", "THEORY"),
    ("Noether's Theorem", "Every continuous symmetry of action corresponds to conservation law", "THEORY"),
    ("Virial Theorem", "Average kinetic energy related to average potential energy for bound systems", "THEORY"),
    ("Principle of Least Action", "Physical system follows path that minimizes action integral", "THEORY"),
    ("Galilean Relativity", "Laws of physics same in all inertial frames, Galilean transformations", "THEORY"),
    # Gravitation & Relativity
    ("General Relativity", "Einstein's theory: gravity is curvature of spacetime caused by mass-energy", "THEORY"),
    ("Special Relativity", "Einstein's theory: speed of light constant in all frames, E=mc^2, time dilation", "THEORY"),
    ("Mass-Energy Equivalence", "E = mc^2, energy and mass are interchangeable", "THEORY"),
    ("Time Dilation", "Moving clocks run slower, t = γt₀ where γ = 1/√(1-v²/c²)", "THEORY"),
    ("Length Contraction", "Moving objects shortened in direction of motion L = L₀/γ", "THEORY"),
    ("Lorentz Transformation", "Coordinate transformations between inertial frames in special relativity", "THEORY"),
    ("Gravitational Time Dilation", "Clocks run slower in stronger gravitational fields", "THEORY"),
    ("Gravitational Lensing", "Light bent by massive objects curving spacetime", "THEORY"),
    ("Gravitational Waves", "Ripples in spacetime from accelerating masses, detected 2015 by LIGO", "FACT"),
    ("Einstein Field Equations", "Gμν + Λgμν = 8πG/c⁴ Tμν, relate spacetime geometry to energy-momentum", "THEORY"),
    ("Schwarzschild Metric", "Solution to Einstein equations for spherically symmetric non-rotating mass, defines black hole", "THEORY"),
    ("Kerr Metric", "Solution for rotating black hole spacetime", "THEORY"),
    ("Equivalence Principle", "Gravitational and inertial mass are equivalent, foundation of general relativity", "THEORY"),
    ("Frame Dragging", "Rotating massive object drags spacetime around it, Lense-Thirring effect", "THEORY"),
    ("Geodesic Equation", "Equation of motion for free particles in curved spacetime", "THEORY"),
    ("Friedmann Equations", "Govern expansion of space in homogeneous isotropic universe", "THEORY"),
    ("Cosmological Constant", "Lambda term in Einstein equations, represents dark energy density", "THEORY"),
    # Electromagnetism
    ("Coulomb's Law", "Electric force between charges F = kq₁q₂/r^2", "THEORY"),
    ("Gauss's Law", "Electric flux through closed surface proportional to enclosed charge", "THEORY"),
    ("Gauss's Law for Magnetism", "Magnetic flux through any closed surface is zero, no magnetic monopoles", "THEORY"),
    ("Faraday's Law of Induction", "Changing magnetic flux induces electromotive force EMF = -dΦ/dt", "THEORY"),
    ("Ampere's Law", "Magnetic field around current proportional to current, with Maxwell correction", "THEORY"),
    ("Maxwell's Equations", "Four equations unifying electricity and magnetism, predict electromagnetic waves", "THEORY"),
    ("Biot-Savart Law", "Magnetic field from current element dB = μ₀/4π (Idl×r̂)/r²", "THEORY"),
    ("Lorentz Force Law", "Force on charge in EM field F = q(E + v×B)", "THEORY"),
    ("Ohm's Law", "Voltage equals current times resistance V = IR", "THEORY"),
    ("Kirchhoff's Current Law", "Sum of currents entering junction equals sum leaving, charge conservation", "THEORY"),
    ("Kirchhoff's Voltage Law", "Sum of voltage drops around closed loop equals zero, energy conservation", "THEORY"),
    ("Joule's Law", "Power dissipated by resistor P = I²R", "THEORY"),
    ("Lenz's Law", "Induced current opposes change in magnetic flux that produced it", "THEORY"),
    ("Poynting Vector", "S = E×B/μ₀, energy flux of electromagnetic field", "DEFINITION"),
    ("Electromagnetic Induction", "Production of EMF by changing magnetic field through conductor", "DEFINITION"),
    ("Skin Effect", "Alternating current concentrates near conductor surface at high frequencies", "DEFINITION"),
    ("Displacement Current", "Maxwell's addition to Ampere's law, changing electric field acts as current", "DEFINITION"),
    # Thermodynamics
    ("Zeroth Law of Thermodynamics", "If A in thermal equilibrium with B and B with C then A with C, defines temperature", "THEORY"),
    ("First Law of Thermodynamics", "Energy conservation: change in internal energy equals heat added minus work done ΔU=Q-W", "THEORY"),
    ("Second Law of Thermodynamics", "Entropy of isolated system never decreases, heat flows from hot to cold spontaneously", "THEORY"),
    ("Third Law of Thermodynamics", "Entropy approaches constant as temperature approaches absolute zero", "THEORY"),
    ("Ideal Gas Law", "PV = nRT relating pressure volume temperature and amount of ideal gas", "THEORY"),
    ("Boyle's Law", "At constant temperature gas pressure inversely proportional to volume P₁V₁=P₂V₂", "THEORY"),
    ("Charles's Law", "At constant pressure gas volume proportional to temperature V/T=constant", "THEORY"),
    ("Gay-Lussac's Law", "At constant volume gas pressure proportional to temperature P/T=constant", "THEORY"),
    ("Avogadro's Law", "Equal volumes of gases at same T and P contain equal number of molecules", "THEORY"),
    ("Dalton's Law of Partial Pressures", "Total pressure of gas mixture equals sum of partial pressures", "THEORY"),
    ("Stefan-Boltzmann Law", "Total radiated power proportional to T^4, P = σAT^4", "THEORY"),
    ("Wien's Displacement Law", "Peak wavelength of blackbody inversely proportional to temperature λmax=b/T", "THEORY"),
    ("Planck's Law", "Spectral radiance of blackbody as function of wavelength and temperature", "THEORY"),
    ("Boltzmann Distribution", "Probability of state proportional to e^(-E/kT), statistical mechanics foundation", "THEORY"),
    ("Maxwell-Boltzmann Distribution", "Speed distribution of particles in ideal gas at thermal equilibrium", "THEORY"),
    ("Carnot's Theorem", "No heat engine more efficient than Carnot engine between same temperatures", "THEORY"),
    ("Clausius Inequality", "Cyclic integral of δQ/T ≤ 0, equality for reversible processes", "THEORY"),
    ("Fourier's Law of Heat Conduction", "Heat flux proportional to negative temperature gradient q = -k∇T", "THEORY"),
    ("Newton's Law of Cooling", "Rate of heat loss proportional to temperature difference with surroundings", "THEORY"),
    ("Equipartition Theorem", "Each quadratic degree of freedom contributes 1/2 kT to average energy", "THEORY"),
    ("Entropy", "Measure of disorder or number of microstates, S = k ln Ω in statistical mechanics", "DEFINITION"),
    # Quantum Mechanics
    ("Schrödinger Equation", "Fundamental equation of quantum mechanics describing wave function evolution iℏ∂ψ/∂t = Ĥψ", "THEORY"),
    ("Heisenberg Uncertainty Principle", "Cannot simultaneously know exact position and momentum ΔxΔp ≥ ℏ/2", "THEORY"),
    ("Pauli Exclusion Principle", "No two identical fermions can occupy same quantum state simultaneously", "THEORY"),
    ("Wave-Particle Duality", "All matter exhibits both wave and particle properties", "THEORY"),
    ("De Broglie Hypothesis", "Every particle has associated wavelength λ = h/p", "THEORY"),
    ("Photoelectric Effect", "Light ejects electrons from metal, energy depends on frequency not intensity", "FACT"),
    ("Compton Scattering", "X-ray photon wavelength increases when scattered by electron", "FACT"),
    ("Born Rule", "Probability of measurement outcome is square of wave function amplitude |ψ|²", "THEORY"),
    ("Superposition Principle (QM)", "Quantum system can exist in multiple states simultaneously until measured", "THEORY"),
    ("Quantum Tunneling", "Particle passing through potential barrier higher than its energy", "THEORY"),
    ("Quantum Entanglement", "Correlated quantum states where measuring one instantly determines the other", "THEORY"),
    ("Bell's Theorem", "No local hidden variable theory can reproduce all quantum mechanical predictions", "THEORY"),
    ("Copenhagen Interpretation", "Wave function collapses upon measurement, no definite state before observation", "THEORY"),
    ("Many-Worlds Interpretation", "All possible quantum measurement outcomes are realized in branching universes", "THEORY"),
    ("Dirac Equation", "Relativistic quantum equation for spin-1/2 particles, predicted antimatter", "THEORY"),
    ("Klein-Gordon Equation", "Relativistic quantum equation for spin-0 particles", "THEORY"),
    ("Spin", "Intrinsic angular momentum of elementary particles, quantized in half-integer or integer units", "DEFINITION"),
    ("Quantum Field Theory", "Framework combining quantum mechanics and special relativity, particles as field excitations", "THEORY"),
    ("Quantum Electrodynamics", "QFT of electromagnetic interaction, most precisely tested theory in physics", "THEORY"),
    ("Quantum Chromodynamics", "QFT of strong interaction between quarks and gluons", "THEORY"),
    ("Renormalization", "Technique to handle infinities in quantum field theory calculations", "DEFINITION"),
    ("Feynman Diagrams", "Pictorial representation of particle interactions in quantum field theory", "DEFINITION"),
    ("Path Integral Formulation", "Feynman's formulation summing over all possible paths", "THEORY"),
    ("Bose-Einstein Condensate", "State of matter at near absolute zero where bosons occupy same quantum state", "DEFINITION"),
    ("Fermi-Dirac Statistics", "Statistical mechanics for identical fermions obeying exclusion principle", "THEORY"),
    ("Bose-Einstein Statistics", "Statistical mechanics for identical bosons that can share quantum states", "THEORY"),
    ("Casimir Effect", "Attractive force between close parallel plates from quantum vacuum fluctuations", "FACT"),
    ("Lamb Shift", "Small energy difference in hydrogen levels from quantum fluctuations, confirmed QED", "FACT"),
    ("Stern-Gerlach Experiment", "Demonstrated quantization of angular momentum with silver atoms in magnetic field", "FACT"),
    ("Double-Slit Experiment", "Demonstrates wave-particle duality, single particles create interference pattern", "FACT"),
    ("Quantum Decoherence", "Loss of quantum coherence through interaction with environment", "THEORY"),
    ("No-Cloning Theorem", "Impossible to create identical copy of arbitrary unknown quantum state", "THEORY"),
    ("Correspondence Principle", "Quantum mechanics reproduces classical physics in limit of large quantum numbers", "THEORY"),
    # Nuclear Physics
    ("Nuclear Fission", "Splitting heavy atomic nucleus into lighter nuclei releasing energy", "DEFINITION"),
    ("Nuclear Fusion", "Combining light atomic nuclei into heavier nucleus releasing energy", "DEFINITION"),
    ("Radioactive Decay", "Spontaneous emission of radiation from unstable atomic nucleus", "DEFINITION"),
    ("Alpha Decay", "Nucleus emits alpha particle (helium-4 nucleus), reduces atomic number by 2", "DEFINITION"),
    ("Beta Decay", "Neutron converts to proton emitting electron and antineutrino, or vice versa", "DEFINITION"),
    ("Gamma Decay", "Excited nucleus emits gamma ray photon, no change in atomic number", "DEFINITION"),
    ("Half-Life", "Time for half of radioactive sample to decay, characteristic of each isotope", "DEFINITION"),
    ("Binding Energy per Nucleon", "Average energy needed to remove nucleon from nucleus, peaks at iron-56", "DEFINITION"),
    ("Strong Nuclear Force (Residual)", "Force binding protons and neutrons in nucleus, residual of color force", "DEFINITION"),
    ("Nuclear Shell Model", "Model of nucleus with nucleons in quantum energy levels like electron shells", "THEORY"),
    ("Liquid Drop Model", "Nuclear model treating nucleus as incompressible fluid drop", "THEORY"),
    ("Bethe-Weizsäcker Formula", "Semi-empirical mass formula for nuclear binding energy", "THEORY"),
    ("Cross Section", "Effective area for nuclear or particle interaction, measured in barns", "DEFINITION"),
    # Atomic Physics
    ("Bohr Model", "Atomic model with quantized electron orbits and energy levels E_n = -13.6/n² eV for hydrogen", "THEORY"),
    ("Rydberg Formula", "Predicts wavelengths of hydrogen spectral lines 1/λ = R(1/n₁² - 1/n₂²)", "THEORY"),
    ("Atomic Orbital", "Wave function describing probability of finding electron around nucleus", "DEFINITION"),
    ("Quantum Numbers", "Set of four numbers (n, l, ml, ms) describing electron state in atom", "DEFINITION"),
    ("Aufbau Principle", "Electrons fill lowest energy orbitals first", "THEORY"),
    ("Hund's Rule", "Electrons fill degenerate orbitals singly before pairing with parallel spins", "THEORY"),
    ("Zeeman Effect", "Splitting of spectral lines in magnetic field", "FACT"),
    ("Stark Effect", "Splitting of spectral lines in electric field", "FACT"),
    ("Fine Structure", "Splitting of spectral lines from spin-orbit coupling and relativistic effects", "DEFINITION"),
    ("Hyperfine Structure", "Small energy splitting from nuclear spin interaction, basis of atomic clocks", "DEFINITION"),
    # Optics
    ("Snell's Law", "n₁sinθ₁ = n₂sinθ₂, relates angles of incidence and refraction at interface", "THEORY"),
    ("Law of Reflection", "Angle of incidence equals angle of reflection", "THEORY"),
    ("Total Internal Reflection", "Light completely reflected at interface when angle exceeds critical angle", "DEFINITION"),
    ("Brewster's Angle", "Angle at which reflected light is completely polarized tan θ_B = n₂/n₁", "THEORY"),
    ("Diffraction", "Bending of waves around obstacles or through apertures", "DEFINITION"),
    ("Interference", "Superposition of waves producing reinforcement or cancellation", "DEFINITION"),
    ("Constructive Interference", "Waves in phase combine to produce larger amplitude", "DEFINITION"),
    ("Destructive Interference", "Waves out of phase cancel each other", "DEFINITION"),
    ("Thin Film Interference", "Interference from reflections at top and bottom of thin film", "DEFINITION"),
    ("Huygens' Principle", "Every point on wavefront is source of secondary spherical wavelets", "THEORY"),
    ("Fresnel Equations", "Describe reflection and transmission of light at interface for each polarization", "THEORY"),
    ("Rayleigh Scattering", "Scattering of light by particles smaller than wavelength, explains blue sky", "THEORY"),
    ("Mie Scattering", "Scattering by particles comparable to wavelength, explains white clouds", "THEORY"),
    ("Raman Scattering", "Inelastic scattering where photon energy shifts by vibrational mode", "DEFINITION"),
    ("Polarization", "Orientation of oscillation plane of transverse wave like light", "DEFINITION"),
    ("Malus's Law", "Intensity through polarizer I = I₀cos²θ", "THEORY"),
    ("Diffraction Grating", "Optical component with periodic structure dispersing light into spectra", "DEFINITION"),
    ("Airy Disk", "Central bright spot of diffraction pattern from circular aperture", "DEFINITION"),
    ("Rayleigh Criterion", "Minimum angular separation for resolving two point sources θ = 1.22λ/D", "THEORY"),
    ("Optical Fiber", "Flexible transparent fiber guiding light by total internal reflection", "DEFINITION"),
    ("Laser", "Light Amplification by Stimulated Emission of Radiation, coherent monochromatic light", "DEFINITION"),
    ("Stimulated Emission", "Photon triggers emission of identical photon from excited atom", "DEFINITION"),
    ("Spontaneous Emission", "Excited atom emits photon randomly without external trigger", "DEFINITION"),
    ("Absorption Spectrum", "Dark lines in continuous spectrum from absorption by cooler gas", "DEFINITION"),
    ("Emission Spectrum", "Bright lines at specific wavelengths from excited atoms", "DEFINITION"),
    ("Blackbody Radiation", "Electromagnetic radiation emitted by perfect absorber in thermal equilibrium", "DEFINITION"),
    ("Cherenkov Radiation", "Light emitted when charged particle moves faster than light in medium", "DEFINITION"),
    ("Synchrotron Radiation", "Radiation from charged particles accelerating in magnetic field", "DEFINITION"),
    ("Bremsstrahlung", "Radiation from deceleration of charged particle by another, braking radiation", "DEFINITION"),
    # Wave Physics
    ("Wave Equation", "∂²u/∂t² = c²∂²u/∂x², describes propagation of waves", "THEORY"),
    ("Doppler Effect", "Change in observed frequency when source and observer in relative motion", "THEORY"),
    ("Redshift", "Increase in wavelength of light from receding source or expanding space", "DEFINITION"),
    ("Blueshift", "Decrease in wavelength of light from approaching source", "DEFINITION"),
    ("Hubble's Law", "Recession velocity of galaxy proportional to distance v = H₀d", "THEORY"),
    ("Standing Wave", "Wave pattern from superposition of two traveling waves in opposite directions", "DEFINITION"),
    ("Resonance", "Large amplitude oscillation when driving frequency matches natural frequency", "DEFINITION"),
    ("Harmonics", "Integer multiples of fundamental frequency in standing wave", "DEFINITION"),
    ("Fourier Analysis", "Decomposition of function into sum of sinusoidal components", "DEFINITION"),
    ("Superposition Principle (Waves)", "Net displacement is sum of individual wave displacements at each point", "THEORY"),
    ("Dispersion", "Dependence of wave speed on frequency, causes spreading of wave packet", "DEFINITION"),
    ("Group Velocity", "Velocity of wave packet envelope, speed of energy/information transport", "DEFINITION"),
    ("Phase Velocity", "Speed of individual wave crest, v_p = ω/k", "DEFINITION"),
    # Acoustics
    ("Speed of Sound", "~343 m/s in air at 20°C, depends on medium properties", "FACT"),
    ("Decibel", "Logarithmic unit of sound intensity, dB = 10 log(I/I₀)", "DEFINITION"),
    ("Acoustic Resonance", "Reinforcement of sound in cavity at natural frequencies", "DEFINITION"),
    ("Doppler Effect (Sound)", "Frequency shift of sound from relative motion of source and observer", "THEORY"),
    ("Sonic Boom", "Shock wave produced when object travels faster than speed of sound", "DEFINITION"),
    ("Mach Number", "Ratio of object speed to speed of sound, Mach 1 = speed of sound", "DEFINITION"),
    ("Ultrasound", "Sound waves above human hearing range >20 kHz, used in medical imaging", "DEFINITION"),
    ("Infrasound", "Sound waves below human hearing range <20 Hz", "DEFINITION"),
    # Cosmology
    ("Big Bang Theory", "Universe began from extremely hot dense state ~13.8 billion years ago and has been expanding", "THEORY"),
    ("Cosmic Microwave Background", "Thermal radiation from early universe ~380000 years after Big Bang, 2.725 K", "FACT"),
    ("Cosmic Inflation", "Exponential expansion of early universe in first ~10^-36 seconds", "THEORY"),
    ("Dark Matter", "Invisible matter making up ~27% of universe, detectable only through gravity", "THEORY"),
    ("Dark Energy", "Mysterious energy causing accelerating expansion of universe, ~68% of universe", "THEORY"),
    ("Hubble Constant", "Rate of expansion of universe, ~70 km/s/Mpc, tension between measurement methods", "FACT"),
    ("Olbers' Paradox", "Why is night sky dark if universe has infinite stars, resolved by finite age and expansion", "THEORY"),
    ("Cosmic Distance Measurement", "Methods to determine astronomical distances: parallax, Cepheids, Type Ia supernovae", "DEFINITION"),
    ("Parallax", "Apparent shift in position of nearby star due to Earth orbital motion, distance measurement", "DEFINITION"),
    ("Parsec", "Distance at which 1 AU subtends 1 arcsecond parallax, ~3.26 light-years", "DEFINITION"),
    ("Light-Year", "Distance light travels in one year, ~9.461 × 10^12 km", "DEFINITION"),
    ("Astronomical Unit", "Mean Earth-Sun distance, ~149.6 million km", "DEFINITION"),
    ("Cosmological Principle", "Universe is homogeneous and isotropic on large scales", "THEORY"),
    ("Nucleosynthesis", "Formation of atomic nuclei, Big Bang produced H He Li, stars produce heavier elements", "THEORY"),
    ("Stellar Nucleosynthesis", "Creation of elements in stellar cores through nuclear fusion", "THEORY"),
    ("Chandrasekhar Limit", "Maximum mass of stable white dwarf ~1.4 solar masses", "THEORY"),
    ("Tolman-Oppenheimer-Volkoff Limit", "Maximum mass of neutron star ~2-3 solar masses", "THEORY"),
    ("Eddington Limit", "Maximum luminosity where radiation pressure balances gravity", "THEORY"),
    ("Roche Limit", "Minimum distance satellite can approach primary without being torn apart by tides", "THEORY"),
    ("Schwarzschild Radius", "Radius of event horizon for non-rotating black hole r_s = 2GM/c²", "DEFINITION"),
    ("Event Horizon", "Boundary around black hole beyond which nothing can escape", "DEFINITION"),
    ("Hawking Radiation", "Theoretical radiation from black holes due to quantum effects near event horizon", "THEORY"),
    ("Information Paradox", "Conflict between quantum mechanics unitarity and black hole information loss", "THEORY"),
    ("Penrose Process", "Extraction of energy from rotating black hole ergosphere", "THEORY"),
    ("Cosmic Web", "Large-scale structure of universe: filaments, walls, voids of galaxy clusters", "FACT"),
    ("Great Attractor", "Gravitational anomaly in intergalactic space attracting local galaxy flow", "FACT"),
    ("Multiverse", "Hypothetical set of multiple universes comprising all of physical reality", "THEORY"),
    ("String Theory", "Theoretical framework where point particles replaced by one-dimensional strings", "THEORY"),
    ("M-Theory", "Unifying framework of five superstring theories in 11 dimensions", "THEORY"),
    ("Loop Quantum Gravity", "Theory quantizing spacetime itself into discrete units", "THEORY"),
    ("Grand Unified Theory", "Hypothetical theory unifying strong, weak, and electromagnetic forces", "THEORY"),
    ("Theory of Everything", "Hypothetical single framework explaining all fundamental forces and particles", "THEORY"),
    ("Anthropic Principle", "Universe parameters appear fine-tuned for conscious life observation", "THEORY"),
    ("Fermi Paradox", "Apparent contradiction between lack of evidence for extraterrestrial civilizations and high probability", "THEORY"),
    ("Drake Equation", "Probabilistic formula estimating number of communicating civilizations in galaxy", "THEORY"),
    # More mechanics/physics
    ("Centripetal Force", "Force directed toward center of circular path, F = mv²/r", "DEFINITION"),
    ("Centrifugal Force", "Pseudo-force in rotating frame pointing outward from rotation axis", "DEFINITION"),
    ("Coriolis Force", "Pseudo-force in rotating frame deflecting moving objects perpendicular to velocity", "DEFINITION"),
    ("Friction", "Force opposing relative motion between surfaces in contact", "DEFINITION"),
    ("Drag Force", "Resistance force on object moving through fluid, F_d = 1/2 ρv²C_dA", "DEFINITION"),
    ("Terminal Velocity", "Maximum velocity of falling object when drag equals gravity", "DEFINITION"),
    ("Simple Harmonic Motion", "Oscillation with restoring force proportional to displacement, sinusoidal motion", "DEFINITION"),
    ("Damped Oscillation", "Oscillation with decreasing amplitude due to energy loss", "DEFINITION"),
    ("Forced Oscillation", "Oscillation driven by external periodic force", "DEFINITION"),
    ("Pendulum", "Weight suspended from pivot swinging freely, period T = 2π√(L/g) for small angles", "DEFINITION"),
    ("Moment of Inertia", "Rotational analog of mass, resistance to angular acceleration I = Σmr²", "DEFINITION"),
    ("Torque", "Rotational force, τ = r × F, causes angular acceleration", "DEFINITION"),
    ("Angular Momentum", "Rotational analog of linear momentum L = Iω or L = r × p", "DEFINITION"),
    ("Precession", "Change in orientation of rotational axis of spinning body due to torque", "DEFINITION"),
    ("Gyroscope", "Device using spinning rotor to maintain orientation, precession effects", "DEFINITION"),
    ("Elastic Collision", "Collision conserving both kinetic energy and momentum", "DEFINITION"),
    ("Inelastic Collision", "Collision conserving momentum but not kinetic energy", "DEFINITION"),
    ("Center of Mass", "Point where weighted mass distribution is centered, system moves as if all mass concentrated there", "DEFINITION"),
    ("Rigid Body Dynamics", "Mechanics of solid bodies that do not deform", "DEFINITION"),
    # Electricity
    ("Electric Field", "Region where electric charge experiences force, E = F/q", "DEFINITION"),
    ("Magnetic Field", "Region where moving charges or magnets experience force", "DEFINITION"),
    ("Electric Potential", "Energy per unit charge at point in electric field, voltage V = W/q", "DEFINITION"),
    ("Capacitance", "Ability to store electric charge C = Q/V, unit farad", "DEFINITION"),
    ("Inductance", "Property opposing change in current, stores energy in magnetic field, unit henry", "DEFINITION"),
    ("Resistance", "Opposition to current flow, R = V/I, unit ohm", "DEFINITION"),
    ("Impedance", "Total opposition to AC current including resistance and reactance Z = R + jX", "DEFINITION"),
    ("RC Circuit", "Circuit with resistor and capacitor, time constant τ = RC", "DEFINITION"),
    ("RL Circuit", "Circuit with resistor and inductor, time constant τ = L/R", "DEFINITION"),
    ("RLC Circuit", "Circuit with resistor inductor capacitor, exhibits resonance", "DEFINITION"),
    ("Electromagnetic Wave", "Self-propagating wave of oscillating electric and magnetic fields, speed c in vacuum", "DEFINITION"),
    ("Electromagnetic Spectrum", "Range of all EM radiation frequencies from radio to gamma rays", "DEFINITION"),
    ("Radio Waves", "EM radiation with longest wavelength >1mm, used in communication", "DEFINITION"),
    ("Microwaves", "EM radiation 1mm to 1m wavelength, used in radar and cooking", "DEFINITION"),
    ("Infrared Radiation", "EM radiation between visible light and microwaves, thermal radiation", "DEFINITION"),
    ("Visible Light", "EM radiation visible to human eye, wavelength 380-700 nm", "DEFINITION"),
    ("Ultraviolet Radiation", "EM radiation shorter than visible light, causes sunburn", "DEFINITION"),
    ("X-rays", "High-energy EM radiation, penetrates soft tissue, used in medical imaging", "DEFINITION"),
    ("Gamma Rays", "Highest energy EM radiation from nuclear decay and cosmic sources", "DEFINITION"),
    # Solid State / Materials
    ("Semiconductor", "Material with conductivity between conductor and insulator, basis of electronics", "DEFINITION"),
    ("Superconductivity", "Zero electrical resistance below critical temperature, Meissner effect", "DEFINITION"),
    ("Hall Effect", "Voltage across conductor perpendicular to current and magnetic field", "DEFINITION"),
    ("Piezoelectric Effect", "Generation of electric charge from mechanical stress in certain crystals", "DEFINITION"),
    ("Thermoelectric Effect", "Direct conversion between temperature difference and electric voltage", "DEFINITION"),
    ("Photoelectric Effect Law", "KEmax = hf - φ, Einstein's equation for photoelectric effect", "THEORY"),
    ("Band Gap", "Energy range in solid where no electron states exist, determines electrical properties", "DEFINITION"),
    ("Fermi Energy", "Highest energy level occupied by electrons at absolute zero", "DEFINITION"),
]
for name, defn, typ in physics_laws:
    C(name, defn, typ, 0.96)

# Physics law interconnections
law_connections = [
    ("Newton's Second Law", "Newton's First Law", "RELATES_TO"),
    ("Newton's Second Law", "Newton's Third Law", "RELATES_TO"),
    ("Newton's Law of Universal Gravitation", "Kepler's Third Law", "ENABLES"),
    ("Conservation of Energy", "First Law of Thermodynamics", "RELATES_TO"),
    ("Conservation of Momentum", "Newton's Third Law", "RELATES_TO"),
    ("Noether's Theorem", "Conservation of Energy", "ENABLES"),
    ("Noether's Theorem", "Conservation of Momentum", "ENABLES"),
    ("Noether's Theorem", "Conservation of Angular Momentum", "ENABLES"),
    ("Maxwell's Equations", "Gauss's Law", "PART_OF"),
    ("Maxwell's Equations", "Gauss's Law for Magnetism", "PART_OF"),
    ("Maxwell's Equations", "Faraday's Law of Induction", "PART_OF"),
    ("Maxwell's Equations", "Ampere's Law", "PART_OF"),
    ("Maxwell's Equations", "Electromagnetic Wave", "ENABLES"),
    ("Maxwell's Equations", "Special Relativity", "ENABLES"),
    ("Special Relativity", "Mass-Energy Equivalence", "ENABLES"),
    ("Special Relativity", "Time Dilation", "ENABLES"),
    ("Special Relativity", "Length Contraction", "ENABLES"),
    ("Special Relativity", "Lorentz Transformation", "RELATES_TO"),
    ("General Relativity", "Gravitational Lensing", "ENABLES"),
    ("General Relativity", "Gravitational Waves", "ENABLES"),
    ("General Relativity", "Black Hole", "ENABLES"),
    ("General Relativity", "Einstein Field Equations", "RELATES_TO"),
    ("General Relativity", "Gravitational Time Dilation", "ENABLES"),
    ("Equivalence Principle", "General Relativity", "ENABLES"),
    ("Schrödinger Equation", "Quantum Tunneling", "ENABLES"),
    ("Schrödinger Equation", "Atomic Orbital", "ENABLES"),
    ("Heisenberg Uncertainty Principle", "Wave-Particle Duality", "RELATES_TO"),
    ("Pauli Exclusion Principle", "Electron", "RELATES_TO"),
    ("Pauli Exclusion Principle", "White Dwarf", "ENABLES"),
    ("De Broglie Hypothesis", "Wave-Particle Duality", "ENABLES"),
    ("Photoelectric Effect", "Quantum Mechanics", "ENABLES"),
    ("Planck's Law", "Blackbody Radiation", "RELATES_TO"),
    ("Stefan-Boltzmann Law", "Blackbody Radiation", "RELATES_TO"),
    ("Wien's Displacement Law", "Blackbody Radiation", "RELATES_TO"),
    ("Quantum Field Theory", "Quantum Electrodynamics", "ENABLES"),
    ("Quantum Field Theory", "Quantum Chromodynamics", "ENABLES"),
    ("Quantum Electrodynamics", "Electromagnetic Force", "RELATES_TO"),
    ("Quantum Chromodynamics", "Strong Nuclear Force", "RELATES_TO"),
    ("Big Bang Theory", "Cosmic Microwave Background", "ENABLES"),
    ("Big Bang Theory", "Cosmic Inflation", "RELATES_TO"),
    ("Big Bang Theory", "Nucleosynthesis", "ENABLES"),
    ("Hubble's Law", "Big Bang Theory", "ENABLES"),
    ("Dark Energy", "Cosmological Constant", "RELATES_TO"),
    ("Friedmann Equations", "Big Bang Theory", "RELATES_TO"),
    ("Snell's Law", "Total Internal Reflection", "ENABLES"),
    ("Doppler Effect", "Redshift", "ENABLES"),
    ("Doppler Effect", "Blueshift", "ENABLES"),
    ("Nuclear Fission", "Alpha Decay", "RELATES_TO"),
    ("Nuclear Fusion", "Stellar Nucleosynthesis", "RELATES_TO"),
    ("Stellar Nucleosynthesis", "Star", "RELATES_TO"),
    ("Chandrasekhar Limit", "White Dwarf", "RELATES_TO"),
    ("Chandrasekhar Limit", "Type Ia Supernova", "ENABLES"),
    ("Bohr Model", "Rydberg Formula", "RELATES_TO"),
    ("Bohr Model", "Quantum Numbers", "ENABLES"),
    ("Hooke's Law", "Simple Harmonic Motion", "ENABLES"),
    ("Ohm's Law", "Kirchhoff's Current Law", "RELATES_TO"),
    ("Ohm's Law", "Kirchhoff's Voltage Law", "RELATES_TO"),
    ("Laser", "Stimulated Emission", "ENABLES"),
    ("Bernoulli's Principle", "Drag Force", "RELATES_TO"),
    ("Hawking Radiation", "Black Hole", "RELATES_TO"),
    ("Hawking Radiation", "Quantum Field Theory", "RELATES_TO"),
    ("String Theory", "Theory of Everything", "RELATES_TO"),
    ("Grand Unified Theory", "Theory of Everything", "RELATES_TO"),
    ("Standard Model", "Quantum Field Theory", "RELATES_TO"),
    ("Carnot's Theorem", "Second Law of Thermodynamics", "RELATES_TO"),
    ("Second Law of Thermodynamics", "Entropy", "RELATES_TO"),
    ("Wave Equation", "Standing Wave", "ENABLES"),
    ("Fourier Analysis", "Harmonics", "RELATES_TO"),
    ("Rayleigh Scattering", "Visible Light", "RELATES_TO"),
]
for src, tgt, rel in law_connections:
    R(src, tgt, rel, 0.9)

# ============================================================
# UNITS OF MEASUREMENT (100+)
# ============================================================
units = [
    # SI Base Units
    ("Meter", "SI base unit of length, defined by speed of light", "DEFINITION"),
    ("Kilogram", "SI base unit of mass, defined by Planck constant since 2019", "DEFINITION"),
    ("Second", "SI base unit of time, defined by cesium-133 atom transitions", "DEFINITION"),
    ("Ampere", "SI base unit of electric current, defined by elementary charge", "DEFINITION"),
    ("Kelvin", "SI base unit of temperature, defined by Boltzmann constant", "DEFINITION"),
    ("Mole", "SI base unit of amount of substance, Avogadro number of entities", "DEFINITION"),
    ("Candela", "SI base unit of luminous intensity", "DEFINITION"),
    # SI Derived Units
    ("Newton", "SI unit of force, 1 N = 1 kg⋅m/s²", "DEFINITION"),
    ("Joule", "SI unit of energy, 1 J = 1 N⋅m = 1 kg⋅m²/s²", "DEFINITION"),
    ("Watt", "SI unit of power, 1 W = 1 J/s", "DEFINITION"),
    ("Pascal", "SI unit of pressure, 1 Pa = 1 N/m²", "DEFINITION"),
    ("Hertz", "SI unit of frequency, 1 Hz = 1 cycle per second", "DEFINITION"),
    ("Coulomb", "SI unit of electric charge, 1 C = 1 A⋅s", "DEFINITION"),
    ("Volt", "SI unit of electric potential, 1 V = 1 J/C", "DEFINITION"),
    ("Ohm", "SI unit of electrical resistance, 1 Ω = 1 V/A", "DEFINITION"),
    ("Farad", "SI unit of capacitance, 1 F = 1 C/V", "DEFINITION"),
    ("Henry", "SI unit of inductance, 1 H = 1 V⋅s/A", "DEFINITION"),
    ("Weber", "SI unit of magnetic flux, 1 Wb = 1 V⋅s", "DEFINITION"),
    ("Tesla", "SI unit of magnetic flux density, 1 T = 1 Wb/m²", "DEFINITION"),
    ("Siemens", "SI unit of electrical conductance, 1 S = 1/Ω", "DEFINITION"),
    ("Lumen", "SI unit of luminous flux", "DEFINITION"),
    ("Lux", "SI unit of illuminance, 1 lx = 1 lm/m²", "DEFINITION"),
    ("Becquerel", "SI unit of radioactivity, 1 Bq = 1 decay per second", "DEFINITION"),
    ("Gray", "SI unit of absorbed radiation dose, 1 Gy = 1 J/kg", "DEFINITION"),
    ("Sievert", "SI unit of equivalent radiation dose, accounts for biological effect", "DEFINITION"),
    ("Radian", "SI unit of plane angle, full circle = 2π radians", "DEFINITION"),
    ("Steradian", "SI unit of solid angle", "DEFINITION"),
    ("Katal", "SI unit of catalytic activity", "DEFINITION"),
    # Non-SI but common
    ("Electronvolt", "Energy unit in particle physics, 1 eV = 1.602×10⁻¹⁹ J", "DEFINITION"),
    ("Angstrom", "Unit of length, 1 Å = 10⁻¹⁰ m, used for atomic scales", "DEFINITION"),
    ("Bar", "Unit of pressure, 1 bar = 100000 Pa", "DEFINITION"),
    ("Atmosphere", "Unit of pressure, 1 atm = 101325 Pa, standard atmospheric pressure", "DEFINITION"),
    ("Calorie", "Unit of energy, 1 cal = 4.184 J, energy to heat 1g water by 1°C", "DEFINITION"),
    ("Kilowatt-hour", "Unit of energy, 1 kWh = 3.6 MJ, common for electricity billing", "DEFINITION"),
    ("Horsepower", "Unit of power, 1 hp ≈ 746 W", "DEFINITION"),
    ("Pound-force", "Imperial unit of force, 1 lbf ≈ 4.448 N", "DEFINITION"),
    ("Mile", "Imperial unit of length, 1 mi = 1609.344 m", "DEFINITION"),
    ("Foot", "Imperial unit of length, 1 ft = 0.3048 m", "DEFINITION"),
    ("Inch", "Imperial unit of length, 1 in = 25.4 mm", "DEFINITION"),
    ("Yard", "Imperial unit of length, 1 yd = 0.9144 m", "DEFINITION"),
    ("Pound Mass", "Imperial unit of mass, 1 lb = 0.4536 kg", "DEFINITION"),
    ("Ounce", "Imperial unit of mass, 1 oz = 28.35 g", "DEFINITION"),
    ("Gallon", "Imperial unit of volume, US gallon = 3.785 L", "DEFINITION"),
    ("Liter", "Metric unit of volume, 1 L = 0.001 m³", "DEFINITION"),
    ("Nautical Mile", "Unit of distance, 1 nmi = 1852 m, one minute of latitude", "DEFINITION"),
    ("Knot", "Unit of speed, 1 knot = 1 nautical mile per hour", "DEFINITION"),
    # Astronomy units
    ("Solar Mass", "Unit of mass in astronomy, mass of Sun ~1.989×10³⁰ kg", "DEFINITION"),
    ("Solar Luminosity", "Unit of luminosity, luminosity of Sun ~3.828×10²⁶ W", "DEFINITION"),
    ("Solar Radius", "Unit of length in astronomy, radius of Sun ~6.957×10⁸ m", "DEFINITION"),
    ("Jupiter Mass", "Unit of mass, ~1.898×10²⁷ kg, used for exoplanets", "DEFINITION"),
    ("Earth Mass", "Unit of mass, ~5.972×10²⁴ kg", "DEFINITION"),
    # Physics units
    ("Planck Length", "Smallest meaningful length ~1.616×10⁻³⁵ m", "DEFINITION"),
    ("Planck Time", "Smallest meaningful time interval ~5.391×10⁻⁴⁴ s", "DEFINITION"),
    ("Planck Mass", "Natural unit of mass ~2.176×10⁻⁸ kg", "DEFINITION"),
    ("Planck Temperature", "Natural unit of temperature ~1.416×10³² K", "DEFINITION"),
    ("Planck Energy", "Natural unit of energy ~1.956×10⁹ J", "DEFINITION"),
    ("Barn", "Unit of area in nuclear physics, 1 b = 10⁻²⁸ m²", "DEFINITION"),
    ("Curie", "Old unit of radioactivity, 1 Ci = 3.7×10¹⁰ Bq", "DEFINITION"),
    ("Roentgen", "Old unit of radiation exposure", "DEFINITION"),
    ("Gauss", "CGS unit of magnetic flux density, 1 G = 10⁻⁴ T", "DEFINITION"),
    ("Dyne", "CGS unit of force, 1 dyn = 10⁻⁵ N", "DEFINITION"),
    ("Erg", "CGS unit of energy, 1 erg = 10⁻⁷ J", "DEFINITION"),
    ("Poise", "CGS unit of dynamic viscosity", "DEFINITION"),
    ("Stokes", "CGS unit of kinematic viscosity", "DEFINITION"),
    ("Jansky", "Unit of spectral flux density in radio astronomy, 1 Jy = 10⁻²⁶ W/m²/Hz", "DEFINITION"),
    ("Magnitude (Apparent)", "Logarithmic measure of brightness of celestial object as seen from Earth", "DEFINITION"),
    ("Magnitude (Absolute)", "Brightness of celestial object at standard distance of 10 parsecs", "DEFINITION"),
    # Temperature scales
    ("Celsius", "Temperature scale, 0°C = water freezing, 100°C = boiling at 1 atm", "DEFINITION"),
    ("Fahrenheit", "Temperature scale, 32°F = water freezing, 212°F = boiling at 1 atm", "DEFINITION"),
    ("Rankine", "Absolute temperature scale based on Fahrenheit degrees", "DEFINITION"),
    # Time units
    ("Minute", "Unit of time, 60 seconds", "DEFINITION"),
    ("Hour", "Unit of time, 3600 seconds", "DEFINITION"),
    ("Day", "Unit of time, 86400 seconds, one Earth rotation", "DEFINITION"),
    ("Year", "Unit of time, ~365.25 days, one Earth orbit around Sun", "DEFINITION"),
    # Misc
    ("Atomic Mass Unit", "Unit of mass, 1 u = 1/12 mass of carbon-12, ~1.661×10⁻²⁷ kg", "DEFINITION"),
    ("Bohr Radius", "Most probable distance of electron from nucleus in hydrogen ground state ~0.529 Å", "DEFINITION"),
    ("Hubble Time", "Reciprocal of Hubble constant, approximate age of universe ~14 Gyr", "DEFINITION"),
]
for name, defn, typ in units:
    C(name, defn, typ, 0.95)
    R(name, "Unit of Measurement", "IS_A", 0.95)

C("Unit of Measurement", "Standard quantity used to express physical quantities", "DEFINITION", 0.99)
C("SI Units", "International System of Units, modern metric system, seven base units", "DEFINITION", 0.99)
C("CGS Units", "Centimeter-gram-second system of units, older metric system", "DEFINITION", 0.97)

si_base = ["Meter","Kilogram","Second","Ampere","Kelvin","Mole","Candela"]
for u in si_base:
    R(u, "SI Units", "PART_OF", 0.98)

# Physics constants
constants = [
    ("Speed of Light", "Fundamental constant c = 299792458 m/s exactly, maximum speed in universe", "FACT"),
    ("Gravitational Constant", "G = 6.674×10⁻¹¹ N⋅m²/kg², strength of gravitational interaction", "FACT"),
    ("Planck Constant", "h = 6.626×10⁻³⁴ J⋅s, fundamental quantum of action", "FACT"),
    ("Reduced Planck Constant", "ℏ = h/2π = 1.055×10⁻³⁴ J⋅s", "FACT"),
    ("Boltzmann Constant", "k = 1.381×10⁻²³ J/K, relates temperature to energy", "FACT"),
    ("Avogadro Constant", "Nₐ = 6.022×10²³ mol⁻¹, number of entities in one mole", "FACT"),
    ("Elementary Charge", "e = 1.602×10⁻¹⁹ C, charge of proton", "FACT"),
    ("Vacuum Permittivity", "ε₀ = 8.854×10⁻¹² F/m, electric constant", "FACT"),
    ("Vacuum Permeability", "μ₀ = 4π×10⁻⁷ H/m, magnetic constant", "FACT"),
    ("Fine-Structure Constant", "α ≈ 1/137.036, dimensionless coupling constant of electromagnetism", "FACT"),
    ("Stefan-Boltzmann Constant", "σ = 5.670×10⁻⁸ W/m²/K⁴", "FACT"),
    ("Rydberg Constant", "R∞ = 1.097×10⁷ m⁻¹, relates to hydrogen spectral lines", "FACT"),
    ("Gas Constant", "R = 8.314 J/mol/K, relates to ideal gas law", "FACT"),
    ("Electron Mass", "mₑ = 9.109×10⁻³¹ kg", "FACT"),
    ("Proton Mass", "mₚ = 1.673×10⁻²⁷ kg", "FACT"),
    ("Neutron Mass", "mₙ = 1.675×10⁻²⁷ kg, slightly more than proton", "FACT"),
]
for name, defn, typ in constants:
    C(name, defn, typ, 0.98)
    R(name, "Physical Constant", "IS_A", 0.97)

C("Physical Constant", "Fundamental fixed value in physics that does not change", "DEFINITION", 0.99)

# Some additional cosmological objects
astro_objects = [
    ("Nebula", "Cloud of gas and dust in space, can be star-forming or remnant", "DEFINITION"),
    ("Planetary Nebula", "Glowing shell of gas expelled by dying low-mass star", "DEFINITION"),
    ("Supernova Remnant", "Expanding shell of gas from supernova explosion", "DEFINITION"),
    ("Orion Nebula", "Nearest massive star-forming region M42, visible to naked eye in Orion", "FACT"),
    ("Crab Nebula", "Supernova remnant M1 in Taurus from 1054 CE supernova, contains pulsar", "FACT"),
    ("Eagle Nebula", "Star-forming region M16, Pillars of Creation photographed by Hubble", "FACT"),
    ("Helix Nebula", "Nearest planetary nebula NGC 7293, Eye of God", "FACT"),
    ("Ring Nebula", "Planetary nebula M57 in Lyra", "FACT"),
    ("Asteroid Belt", "Region between Mars and Jupiter containing rocky bodies", "FACT"),
    ("Kuiper Belt", "Region beyond Neptune containing icy bodies including Pluto", "FACT"),
    ("Oort Cloud", "Hypothetical spherical shell of icy objects surrounding Solar System at ~50000-100000 AU", "THEORY"),
    ("Comet", "Icy body that develops coma and tail when approaching Sun", "DEFINITION"),
    ("Asteroid", "Rocky body orbiting Sun, smaller than planet, mostly in asteroid belt", "DEFINITION"),
    ("Meteor", "Streak of light from space debris burning in Earth atmosphere", "DEFINITION"),
    ("Meteorite", "Space rock that survives passage through atmosphere and reaches ground", "DEFINITION"),
    ("Exoplanet", "Planet orbiting star other than the Sun, thousands discovered", "DEFINITION"),
    ("Hot Jupiter", "Gas giant exoplanet orbiting very close to its star", "DEFINITION"),
    ("Habitable Zone", "Region around star where liquid water could exist on planet surface", "DEFINITION"),
    ("Pulsar Wind Nebula", "Nebula powered by pulsar relativistic wind", "DEFINITION"),
    ("Accretion Disk", "Disk of material spiraling into massive object like black hole", "DEFINITION"),
    ("Gamma-Ray Burst", "Most energetic events in universe, brief intense gamma-ray flashes", "DEFINITION"),
    ("Magnetosphere", "Region around planet dominated by its magnetic field", "DEFINITION"),
    ("Van Allen Radiation Belts", "Zones of charged particles trapped by Earth magnetic field", "FACT"),
    ("Aurora", "Light display from charged particles exciting atmospheric gases at magnetic poles", "DEFINITION"),
    ("Solar Wind", "Stream of charged particles ejected from Sun upper atmosphere", "DEFINITION"),
    ("Solar Flare", "Sudden flash of increased brightness on Sun from magnetic reconnection", "DEFINITION"),
    ("Coronal Mass Ejection", "Large expulsion of plasma and magnetic field from solar corona", "DEFINITION"),
    ("Sunspot", "Temporary dark area on Sun surface caused by magnetic flux concentration", "DEFINITION"),
    ("Tidal Locking", "Rotation period equals orbital period so same face always toward partner", "DEFINITION"),
    ("Lagrange Points", "Five equilibrium points in two-body gravitational system", "DEFINITION"),
]
for name, defn, typ in astro_objects:
    C(name, defn, typ, 0.95)

# More relations for astro objects
R("Planetary Nebula", "Nebula", "IS_A", 0.98)
R("Supernova Remnant", "Nebula", "IS_A", 0.95)
R("Orion Nebula", "Nebula", "IS_A", 0.98)
R("Crab Nebula", "Supernova Remnant", "IS_A", 0.97)
R("Eagle Nebula", "Nebula", "IS_A", 0.97)
R("Asteroid Belt", "Solar System", "PART_OF", 0.98)
R("Kuiper Belt", "Solar System", "PART_OF", 0.98)
R("Oort Cloud", "Solar System", "PART_OF", 0.95)
R("Solar Wind", "Sun", "PART_OF", 0.9)
R("Solar Wind", "Aurora", "CAUSES", 0.9)
R("Solar Flare", "Coronal Mass Ejection", "CAUSES", 0.85)
R("Coronal Mass Ejection", "Aurora", "CAUSES", 0.9)
R("Van Allen Radiation Belts", "Earth", "PART_OF", 0.95)
R("Magnetosphere", "Earth", "PART_OF", 0.9)
R("Moon", "Tidal Locking", "HAS_PROPERTY", 0.95)
R("Accretion Disk", "Black Hole", "RELATES_TO", 0.9)
R("Accretion Disk", "Quasar", "ENABLES", 0.9)
R("Exoplanet", "Habitable Zone", "RELATES_TO", 0.85)
R("Kepler Space Telescope", "Exoplanet", "RELATES_TO", 0.93)
R("James Webb Space Telescope", "Infrared Radiation", "USED_IN", 0.9)
R("Hubble Space Telescope", "Visible Light", "USED_IN", 0.9)
R("Chandra X-ray Observatory", "X-rays", "USED_IN", 0.9)
R("Fermi Gamma-ray Space Telescope", "Gamma Rays", "USED_IN", 0.9)

# EM spectrum relations
em_spectrum = ["Radio Waves","Microwaves","Infrared Radiation","Visible Light","Ultraviolet Radiation","X-rays","Gamma Rays"]
for e in em_spectrum:
    R(e, "Electromagnetic Spectrum", "PART_OF", 0.97)
    R(e, "Electromagnetic Wave", "IS_A", 0.95)

# Additional cross-domain relations
extra_relations = [
    ("Nuclear Fusion", "Sun", "ENABLES", 0.95),
    ("Nuclear Fusion", "Star", "ENABLES", 0.95),
    ("Gravitational Force", "Planet", "ENABLES", 0.9),
    ("Gravitational Force", "Star", "ENABLES", 0.9),
    ("Gravitational Force", "Galaxy", "ENABLES", 0.9),
    ("Electromagnetic Force", "Chemical Energy", "ENABLES", 0.9),
    ("Electromagnetic Force", "Laser", "ENABLES", 0.85),
    ("Strong Nuclear Force", "Nuclear Energy", "ENABLES", 0.95),
    ("Strong Nuclear Force", "Nuclear Fusion", "ENABLES", 0.95),
    ("Weak Nuclear Force", "Beta Decay", "ENABLES", 0.95),
    ("Weak Nuclear Force", "Radioactive Decay", "ENABLES", 0.9),
    ("Radioactive Decay", "Half-Life", "HAS_PROPERTY", 0.95),
    ("Alpha Decay", "Radioactive Decay", "IS_A", 0.98),
    ("Beta Decay", "Radioactive Decay", "IS_A", 0.98),
    ("Gamma Decay", "Radioactive Decay", "IS_A", 0.98),
    ("Semiconductor", "Band Gap", "HAS_PROPERTY", 0.93),
    ("Superconductivity", "Resistance", "INHIBITS", 0.95),
    ("Dark Matter", "Galaxy", "ENABLES", 0.85),
    ("Dark Energy", "Cosmological Constant", "RELATES_TO", 0.88),
    ("Cosmic Web", "Galaxy", "RELATES_TO", 0.88),
    ("Parallax", "Parsec", "ENABLES", 0.93),
    ("Hubble's Law", "Redshift", "RELATES_TO", 0.93),
    ("Hubble's Law", "Hubble Constant", "RELATES_TO", 0.95),
    ("Quantum Entanglement", "Bell's Theorem", "RELATES_TO", 0.92),
    ("Quantum Tunneling", "Nuclear Fusion", "ENABLES", 0.88),
    ("Higgs Mechanism", "Higgs Boson", "RELATES_TO", 0.95),
    ("Standard Model", "Quark", "PART_OF", 0.95),
    ("Standard Model", "Lepton", "PART_OF", 0.95),
    ("Standard Model", "Boson", "PART_OF", 0.95),
    ("Dirac Equation", "Antimatter", "ENABLES", 0.92),
    ("Positron", "Antimatter", "IS_A", 0.98),
    ("Antiproton", "Antimatter", "IS_A", 0.98),
    ("Pion", "Meson", "IS_A", 0.98),
    ("Kaon", "Meson", "IS_A", 0.98),
    ("Bose-Einstein Condensate", "Bose-Einstein Statistics", "RELATES_TO", 0.9),
    ("Interference", "Double-Slit Experiment", "RELATES_TO", 0.93),
    ("Diffraction", "Huygens' Principle", "RELATES_TO", 0.9),
    ("Resonance", "Standing Wave", "RELATES_TO", 0.9),
    ("Simple Harmonic Motion", "Pendulum", "RELATES_TO", 0.9),
    ("Damped Oscillation", "Simple Harmonic Motion", "RELATES_TO", 0.88),
    ("Sonic Boom", "Mach Number", "RELATES_TO", 0.9),
    ("Centripetal Force", "Newton's Second Law", "RELATES_TO", 0.9),
    ("Coriolis Force", "Earth", "RELATES_TO", 0.85),
    ("Terminal Velocity", "Drag Force", "RELATES_TO", 0.92),
    ("RLC Circuit", "Resonance", "RELATES_TO", 0.88),
    ("Optical Fiber", "Total Internal Reflection", "ENABLES", 0.93),
    ("Cherenkov Radiation", "Speed of Light", "RELATES_TO", 0.88),
    ("International Space Station", "Low Earth Orbit", "RELATES_TO", 0.93),
    ("Lagrange Points", "James Webb Space Telescope", "RELATES_TO", 0.9),
    ("Gravitational Waves", "LIGO", "RELATES_TO", 0.95),
]

C("Higgs Mechanism", "Process by which particles acquire mass through interaction with Higgs field", "THEORY", 0.96)
C("Quantum Mechanics", "Fundamental theory of physics describing nature at atomic and subatomic scales", "THEORY", 0.99)
C("Low Earth Orbit", "Orbit around Earth at altitude 160-2000 km", "DEFINITION", 0.97)
C("LIGO", "Laser Interferometer Gravitational-Wave Observatory, detected gravitational waves 2015", "FACT", 0.97)

for src, tgt, rel, w in extra_relations:
    R(src, tgt, rel, w)

# ============================================================
# WRITE OUTPUT
# ============================================================
output = {
    "concepts": concepts,
    "relations": relations
}

with open("/home/hirschpekf/brain19/data/wave10_space_physics.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Concepts: {len(concepts)}")
print(f"Relations: {len(relations)}")
print(f"Total entries: {len(concepts) + len(relations)}")
