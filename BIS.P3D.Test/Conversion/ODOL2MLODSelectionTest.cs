using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Xunit;
using BIS.Core.Math;
using BIS.P3D.Conversion;
using BIS.P3D.MLOD;
using Mlod = BIS.P3D.MLOD.MLOD;

namespace BIS.P3D.Test.Conversion
{
    /// <summary>
    /// Covers the parts of ODOL2MLOD that make a debinarised model binarisable again:
    /// named selections, proxy triangles and geometry mass.
    /// </summary>
    public class ODOL2MLODSelectionTest
    {
        private static readonly Vector3P Normal = new Vector3P(0, 0, 1);
        private static readonly (int Bone, byte Weight)[] Unskinned = Array.Empty<(int, byte)>();

        private static Point P(float x, float y, float z) => new Point(new Vector3P(x, y, z), PointFlags.NONE);

        private static Face Tri(int a, int b, int c, string texture = "tex.paa")
        {
            return new Face(3, new[] { new Vertex(a, 0, 0, 0), new Vertex(b, 0, 1, 0), new Vertex(c, 0, 0, 1), new Vertex(0, 0, 0, 0) },
                FaceFlags.DEFAULT, texture, "");
        }

        private static P3DM_LOD Lod(float resolution, Point[] points, Face[] faces, params Tagg[] taggs)
        {
            var normals = points.Select(_ => Normal).ToArray();
            return new P3DM_LOD(resolution, points, normals, faces, taggs.Append(new EOFTagg()));
        }

        private static Mlod Roundtrip(params P3DM_LOD[] lods) => ODOL2MLOD.Convert(MLOD2ODOL.Convert(new Mlod(lods)));

        private static void AssertAt(Vector3P actual, float x, float y, float z)
        {
            Assert.Equal(x, actual.X, 4);
            Assert.Equal(y, actual.Y, 4);
            Assert.Equal(z, actual.Z, 4);
        }

        [Fact]
        public void Roundtrip_SelectionWithVertexList_KeepsPointsAndFaces()
        {
            var points = new[] { P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(1, 1, 0) };
            var faces = new[] { Tri(0, 1, 2), Tri(1, 3, 2) };
            var door = new NamedSelectionTagg("door", new byte[] { 1, 1, 1, 0 }, new byte[] { 1, 0 });

            var result = Roundtrip(Lod(1f, points, faces, door));

            var tagg = result.Lods[0].Taggs.OfType<NamedSelectionTagg>().Single(t => t.Name == "door");
            Assert.Equal(new byte[] { 1, 1, 1, 0 }, tagg.Points);
            Assert.Equal(new byte[] { 1, 0 }, tagg.Faces);
        }

        [Fact]
        public void Roundtrip_GeometryLod_SpreadsTotalMassEvenly()
        {
            var points = new[] { P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(1, 1, 0) };
            var faces = new[] { Tri(0, 1, 2), Tri(1, 3, 2) };

            var result = Roundtrip(Lod(1f, points, faces), Lod(1e13f, points, faces));

            Assert.Empty(result.Lods[0].Taggs.OfType<MassTagg>());
            var mass = result.Lods[1].Taggs.OfType<MassTagg>().Single().Mass;
            Assert.Equal(4, mass.Length);
            // MLOD2ODOL writes a total mass of 1.
            Assert.All(mass, m => Assert.Equal(0.25f, m, 5));
        }

        [Theory]
        [InlineData(255, 1)]
        [InlineData(0, 0)]
        public void EncodeWeight_FullAndNone(byte odol, byte mlod)
        {
            Assert.Equal(mlod, ODOL2MLOD.EncodeWeight(odol));
        }

        [Fact]
        public void EncodeWeight_PartialWeightsDecodeBack()
        {
            for (int odol = 1; odol < 255; odol++)
            {
                byte mlod = ODOL2MLOD.EncodeWeight((byte)odol);
                float decoded = (255 - mlod) / 254f;
                Assert.InRange(decoded, odol / 255f - 0.005f, odol / 255f + 0.005f);
            }
        }

        [Fact]
        public void ProxyCorners_RowsAreAsideUpDirPosition()
        {
            // Yawed 90 degrees: aside along -Z, up along +Y, direction along +X; placed at (1, 2, 3).
            var m = new Matrix4x4(
                0, 0, -1, 0,
                0, 1, 0, 0,
                1, 0, 0, 0,
                1, 2, 3, 1);

            var (origin, up, dir) = ODOL2MLOD.ProxyCorners(m);

            AssertAt(origin, 1, 2, 3);
            AssertAt(up, 1, 4, 3);
            AssertAt(dir, 2, 2, 3);
        }

        [Fact]
        public void MatchProxyFaces_SharedTransform_ResolvedByBone()
        {
            // Two identical proxy triangles at the origin, skinned to different bones.
            var points = new[] { P(0, 0, 0), P(0, 2, 0), P(0, 0, 1), P(0, 0, 0), P(0, 2, 0), P(0, 0, 1) };
            var faces = new[] { Tri(0, 1, 2, ""), Tri(3, 4, 5, "") };
            var bones = new[]
            {
                new[] { (7, (byte)255) }, new[] { (7, (byte)255) }, new[] { (7, (byte)255) },
                new[] { (3, (byte)255) }, new[] { (3, (byte)255) }, new[] { (3, (byte)255) },
            };
            var proxies = new[]
            {
                new ODOL2MLOD.ProxyRef(10, Matrix4x4.Identity, 3),
                new ODOL2MLOD.ProxyRef(11, Matrix4x4.Identity, 7),
            };

            var result = ODOL2MLOD.MatchProxyFaces(proxies, faces, points, bones);

            Assert.Equal(1, result[10]);
            Assert.Equal(0, result[11]);
        }

        [Fact]
        public void MatchProxyFaces_ProxyWithoutBone_TakesUnskinnedFace()
        {
            var points = new[] { P(0, 0, 0), P(0, 2, 0), P(0, 0, 1), P(0, 0, 0), P(0, 2, 0), P(0, 0, 1) };
            var faces = new[] { Tri(0, 1, 2, ""), Tri(3, 4, 5, "") };
            var bones = new[]
            {
                new[] { (5, (byte)255) }, new[] { (5, (byte)255) }, new[] { (5, (byte)255) },
                Unskinned, Unskinned, Unskinned,
            };

            var result = ODOL2MLOD.MatchProxyFaces(new[] { new ODOL2MLOD.ProxyRef(4, Matrix4x4.Identity, -1) }, faces, points, bones);

            Assert.Equal(1, result[4]);
        }

        [Fact]
        public void MatchProxyFaces_SkipsTexturedFacesAndLeavesUnmatchedOut()
        {
            var points = new[] { P(0, 0, 0), P(0, 2, 0), P(0, 0, 1) };
            var faces = new[] { Tri(0, 1, 2, "body.paa") };
            var bones = new[] { Unskinned, Unskinned, Unskinned };
            var moved = Matrix4x4.CreateTranslation(5, 0, 0);

            var result = ODOL2MLOD.MatchProxyFaces(new[]
            {
                new ODOL2MLOD.ProxyRef(0, Matrix4x4.Identity, -1),
                new ODOL2MLOD.ProxyRef(1, moved, -1),
            }, faces, points, bones);

            Assert.Empty(result);
        }

        [Fact]
        public void MatchProxyFaces_UsesEachFaceOnce()
        {
            var points = new[] { P(0, 0, 0), P(0, 2, 0), P(0, 0, 1) };
            var faces = new[] { Tri(0, 1, 2, "") };
            var bones = new[] { Unskinned, Unskinned, Unskinned };

            var result = ODOL2MLOD.MatchProxyFaces(new[]
            {
                new ODOL2MLOD.ProxyRef(0, Matrix4x4.Identity, -1),
                new ODOL2MLOD.ProxyRef(1, Matrix4x4.Identity, -1),
            }, faces, points, bones);

            Assert.Single(result);
            Assert.Equal(0, result[0]);
        }

        [Fact]
        public void GroupVerticesByBone_UsesBoneNamesCaseInsensitively()
        {
            var bones = new[]
            {
                new[] { (0, (byte)255) },
                new[] { (1, (byte)128), (0, (byte)127) },
                Unskinned,
            };

            var groups = ODOL2MLOD.GroupVerticesByBone(bones, new[] { "main rotor", "door" });

            Assert.Equal(new List<(int, byte)> { (0, 255), (1, 127) }, groups["MAIN ROTOR"]);
            Assert.Equal(new List<(int, byte)> { (1, 128) }, groups["door"]);
            Assert.Equal(2, groups.Count);
        }
    }
}
