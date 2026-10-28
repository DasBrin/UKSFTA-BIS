using System;
using System.Collections.Generic;
using System.Linq;
using BIS.Core;
using BIS.Core.Math;
using BIS.P3D.ODOL;
using BIS.P3D.MLOD;

namespace BIS.P3D.Conversion
{
    public static class ODOL2MLOD
    {
        /// <summary>ODOL version threshold for 32-bit (vs 16-bit) face indices.</summary>
        private const int LongFaceIndicesVersion = 69;

        private const float GeometryResolution = 1e13f;

        // Proxy triangle legs: the long leg points along the proxy's up axis, the
        // short leg along its direction axis, with the right angle at the origin.
        private const float ProxyUpLength = 2f;
        private const float ProxyDirLength = 1f;

        public static BIS.P3D.MLOD.MLOD Convert(BIS.P3D.ODOL.ODOL odol)
        {
            var lods = new List<BIS.P3D.MLOD.P3DM_LOD>();
            var boneNames = odol.ModelInfo?.Skeleton?.SkeletonBoneNames?.Select(b => b.BoneName).ToArray() ?? Array.Empty<string>();
            var massArray = GetMassArray(odol);

            foreach (var lod in odol.Lods)
            {
                try
                {
                    lods.Add(ConvertLod(lod, odol.Version, lod.Resolution == GeometryResolution ? massArray : null, boneNames));
                }
                catch (Exception ex)
                {
                    Terminal.Muted($"Failed converting LOD: {ex.Message}");
                    throw;
                }
            }

            return new BIS.P3D.MLOD.MLOD(lods.ToArray());
        }

        /// <summary>
        /// Per-point masses for the geometry LOD. Binarize usually keeps only the total mass and
        /// centre of mass, in which case the total is spread evenly over the geometry points.
        /// </summary>
        internal static float[]? GetMassArray(BIS.P3D.ODOL.ODOL odol)
        {
            var info = odol.ModelInfo;
            if (info == null) return null;
            if (info.MassArray != null && info.MassArray.Length > 0) return info.MassArray;

            var geometry = odol.Lods.FirstOrDefault(l => l.Resolution == GeometryResolution);
            if (geometry == null || geometry.Vertices.Count == 0 || info.Mass <= 0) return null;

            int count = geometry.Vertices.Count;
            var com = info.CenterOfMass;
            Terminal.Muted($"Per-point mass not stored: spreading {info.Mass:F1} kg over {count} geometry points. " +
                $"Centre of mass ({com.X:F3}, {com.Y:F3}, {com.Z:F3}) becomes " +
                $"({geometry.Vertices.Average(v => v.X):F3}, {geometry.Vertices.Average(v => v.Y):F3}, {geometry.Vertices.Average(v => v.Z):F3})");
            return Enumerable.Repeat(info.Mass / count, count).ToArray();
        }

        private static BIS.P3D.MLOD.P3DM_LOD ConvertLod(BIS.P3D.ODOL.LOD odolLod, int odolVersion, float[]? massArray, string[] boneNames)
        {
            var vertices = odolLod.Vertices;

            // Map Normals (handle both normal and compressed)
            Vector3P[] normals;
            if (odolLod.Normals != null && odolLod.Normals.Count > 0)
            {
                normals = odolLod.Normals.Select(n => n).ToArray();
            }
            else if (odolLod.NormalsCompressed != null && odolLod.NormalsCompressed.Count > 0)
            {
                normals = odolLod.NormalsCompressed.Select(n => (Vector3P)n).ToArray();
            }
            else
            {
                normals = new Vector3P[0];
            }

            var points = new Point[vertices.Count];
            for (int i = 0; i < vertices.Count; i++)
            {
                points[i] = new Point(vertices[i], PointFlags.NONE);
            }

            var (faces, odolFaceIndexMap, sectionIndexMap) = ConvertFaces(odolLod, odolVersion, normals);
            int odolPointCount = points.Length;
            var vertexBones = ReadVertexBones(odolLod, odolPointCount);
            var proxyTriangles = MapProxyTriangles(odolLod, vertexBones, ref points, ref normals, ref faces);

            var taggs = new List<Tagg>();

            // Map UV Sets
            if (odolLod.UvSets != null)
            {
                for (int i = 0; i < odolLod.UvSets.Length; i++)
                {
                    var uvs = odolLod.UvSets[i].GetUV();
                    var faceUVs = new float[faces.Length][,];
                    for (int f = 0; f < faces.Length; f++)
                    {
                        faceUVs[f] = new float[faces[f].VertexCount, 2];
                        for (int v = 0; v < faces[f].VertexCount; v++)
                        {
                            var vIdx = faces[f].Vertices[v].PointIndex;
                            if (uvs != null && vIdx < uvs.Length)
                            {
                                faceUVs[f][v, 0] = uvs[vIdx].X;
                                faceUVs[f][v, 1] = uvs[vIdx].Y;
                            }
                        }
                    }
                    taggs.Add(new UVSetTagg((uint)(faceUVs.Sum(f => f.Length * 4) + 4), i, faceUVs));
                }
            }

            if (odolLod.NamedSelections != null)
            {
                var boneVertices = GroupVerticesByBone(vertexBones, boneNames);
                for (int s = 0; s < odolLod.NamedSelections.Length; s++)
                {
                    var selection = odolLod.NamedSelections[s];
                    var pointsSelection = new byte[points.Length];
                    var facesSelection = new byte[faces.Length];

                    if (proxyTriangles.TryGetValue(s, out var proxyFace))
                    {
                        // A proxy selection must contain exactly its triangle, nothing else.
                        foreach (var v in faces[proxyFace].RealVertices) pointsSelection[v.PointIndex] = 1;
                        facesSelection[proxyFace] = 1;
                        taggs.Add(new NamedSelectionTagg(selection.Name, pointsSelection, facesSelection));
                        continue;
                    }

                    if (selection.SelectedVertices != null && selection.SelectedVertices.Count > 0)
                    {
                        var weights = selection.SelectedVerticesWeights;
                        for (int k = 0; k < selection.SelectedVertices.Count; k++)
                        {
                            int v = selection.SelectedVertices[k];
                            if (v >= 0 && v < odolPointCount)
                                pointsSelection[v] = EncodeWeight(weights != null && k < weights.Count ? weights[k] : (byte)255);
                        }
                    }
                    else if (boneVertices.TryGetValue(selection.Name, out var skinned))
                    {
                        // Skinned LODs keep no vertex list; the bone of the same name has the vertices.
                        foreach (var (v, weight) in skinned) pointsSelection[v] = EncodeWeight(weight);
                    }

                    if (selection.SelectedFaces != null && selection.SelectedFaces.Count > 0)
                    {
                        var faceSet = new HashSet<int>(selection.SelectedFaces);
                        for (int i = 0; i < odolFaceIndexMap.Length; i++)
                            if (faceSet.Contains(odolFaceIndexMap[i])) facesSelection[i] = 1;
                    }
                    else if (selection.IsSectional && selection.Sections != null && selection.Sections.Count > 0)
                    {
                        var sectionSet = new HashSet<int>(selection.Sections);
                        for (int i = 0; i < sectionIndexMap.Length; i++)
                            if (sectionSet.Contains(sectionIndexMap[i])) facesSelection[i] = 1;
                    }

                    // Section-only selections (e.g. hiddenSelections) carry no vertex list: use the faces' points.
                    if (!pointsSelection.Any(b => b != 0))
                    {
                        for (int f = 0; f < facesSelection.Length; f++)
                        {
                            if (facesSelection[f] == 0) continue;
                            foreach (var v in faces[f].RealVertices)
                                if (v.PointIndex < odolPointCount) pointsSelection[v.PointIndex] = 1;
                        }
                    }

                    taggs.Add(new NamedSelectionTagg(selection.Name, pointsSelection, facesSelection));
                }
            }

            if (massArray != null)
            {
                if (massArray.Length != odolPointCount)
                    Terminal.Warning($"Mass array has {massArray.Length} entries but the geometry LOD has {odolPointCount} points");
                var mass = new float[points.Length];
                Array.Copy(massArray, mass, Math.Min(massArray.Length, odolPointCount));
                taggs.Add(new MassTagg(mass));
            }

            // Map Properties
            if (odolLod.NamedProperties != null)
            {
                foreach (var prop in odolLod.NamedProperties)
                {
                    taggs.Add(new PropertyTagg(prop.Item1, prop.Item2));
                }
            }

            // Must add EndOfFile tagg for MLOD validity
            taggs.Add(new EOFTagg());

            return new BIS.P3D.MLOD.P3DM_LOD(odolLod.Resolution, points, normals, faces, taggs);
        }

        /// <summary>
        /// Converts ODOL faces to MLOD faces, triangulating n-gons (5+ vertices).
        /// Returns the MLOD faces plus, per MLOD face, its original ODOL face index and
        /// its section index (for named selection mapping).
        /// </summary>
        private static (Face[] Faces, int[] OdolFaceIndex, int[] SectionIndex) ConvertFaces(BIS.P3D.ODOL.LOD odolLod, int odolVersion, Vector3P[] normals)
        {
            var mlodFaces = new List<Face>();
            var odolIndexMap = new List<int>();
            var sectionIndexMap = new List<int>();

            if (odolLod.Sections == null) return (mlodFaces.ToArray(), odolIndexMap.ToArray(), sectionIndexMap.ToArray());
            if (odolLod.Polygons == null) return (mlodFaces.ToArray(), odolIndexMap.ToArray(), sectionIndexMap.ToArray());

            var allFaces = odolLod.Polygons.Faces;
            var uvs = (odolLod.UvSets != null && odolLod.UvSets.Length > 0) ? odolLod.UvSets[0].GetUV() : null;

            bool isShortFaceIndices = odolVersion < LongFaceIndicesVersion;
            uint sizeOfFace3 = isShortFaceIndices ? 8u : 16u;
            uint padOfFace4 = isShortFaceIndices ? 2u : 4u;

            for (int sectionIdx = 0; sectionIdx < odolLod.Sections.Length; sectionIdx++)
            {
                var section = odolLod.Sections[sectionIdx];
                uint position = 0u;
                for (int faceIdx = 0; faceIdx < allFaces.Length && position < section.FaceUpperIndex; faceIdx++)
                {
                    if (position >= section.FaceLowerIndex && position < section.FaceUpperIndex)
                    {
                        var odolFace = allFaces[faceIdx];
                        var convFaces = ConvertOneFace(odolFace, section, odolLod, uvs, normals);
                        foreach (var mf in convFaces)
                        {
                            mlodFaces.Add(mf);
                            odolIndexMap.Add(faceIdx);
                            sectionIndexMap.Add(sectionIdx);
                        }
                    }
                    position += sizeOfFace3;
                    if (allFaces[faceIdx].VertexIndices.Length == 4)
                        position += padOfFace4;
                }
            }

            return (mlodFaces.ToArray(), odolIndexMap.ToArray(), sectionIndexMap.ToArray());
        }

        /// <summary>
        /// Reads which skeleton bones each vertex is skinned to. Skinned LODs keep no vertex
        /// list per named selection: membership lives in <c>VertexBoneRef</c> as sub-skeleton
        /// indices, mapped to skeleton bones via <c>SubSkeletonsToSkeleton</c>.
        /// </summary>
        /// <returns>Per vertex, its (skeleton bone index, weight) pairs; empty if unskinned.</returns>
        private static (int Bone, byte Weight)[][] ReadVertexBones(BIS.P3D.ODOL.LOD odolLod, int pointCount)
        {
            var result = new (int Bone, byte Weight)[pointCount][];
            var refs = odolLod.VertexBoneRef;
            var subToBone = odolLod.SubSkeletonsToSkeleton;
            for (int v = 0; v < pointCount; v++)
            {
                if (refs == null || subToBone == null || v >= refs.Count)
                {
                    result[v] = Array.Empty<(int, byte)>();
                    continue;
                }
                var r = refs[v];
                var bones = new List<(int, byte)>();
                for (int j = 0; j < Math.Min(r.Count, 4); j++)
                {
                    int sub = r.Data[2 * j];
                    if (sub < subToBone.Length && subToBone[sub] >= 0)
                        bones.Add((subToBone[sub], r.Data[2 * j + 1]));
                }
                result[v] = bones.ToArray();
            }
            return result;
        }

        /// <summary>
        /// Groups skinned vertices by bone name. Bone names are the model.cfg selection
        /// names, so this recovers the vertices of selections that animate.
        /// </summary>
        internal static Dictionary<string, List<(int Vertex, byte Weight)>> GroupVerticesByBone((int Bone, byte Weight)[][] vertexBones, string[] boneNames)
        {
            var result = new Dictionary<string, List<(int, byte)>>(StringComparer.OrdinalIgnoreCase);
            for (int v = 0; v < vertexBones.Length; v++)
            {
                foreach (var (bone, weight) in vertexBones[v])
                {
                    if (bone >= boneNames.Length) continue;
                    if (!result.TryGetValue(boneNames[bone], out var list))
                        result[boneNames[bone]] = list = new List<(int, byte)>();
                    list.Add((v, weight));
                }
            }
            return result;
        }

        /// <summary>
        /// Converts an ODOL selection weight (0..255, 255 = full) to the MLOD encoding
        /// (1 = full, 0 = not selected, otherwise 255 - 254 * weight).
        /// </summary>
        internal static byte EncodeWeight(byte odolWeight)
        {
            if (odolWeight >= 255) return 1;
            if (odolWeight == 0) return 0;
            return (byte)Math.Round(255f - 254f * (odolWeight / 255f));
        }

        /// <summary>A proxy as needed for matching: its selection, transform and bone (-1 if none).</summary>
        internal readonly record struct ProxyRef(int Selection, System.Numerics.Matrix4x4 Transform, int Bone);

        /// <summary>
        /// Finds the triangle of each proxy and adds one for proxies whose triangle is gone.
        /// Binarize keeps each proxy triangle as an untextured face next to the proxy transform.
        /// </summary>
        /// <returns>Named selection index of each proxy to the MLOD face of its triangle.</returns>
        private static Dictionary<int, int> MapProxyTriangles(BIS.P3D.ODOL.LOD odolLod, (int Bone, byte Weight)[][] vertexBones,
            ref Point[] points, ref Vector3P[] normals, ref Face[] faces)
        {
            var proxies = odolLod.RawProxies;
            if (proxies == null || proxies.Length == 0 || odolLod.NamedSelections == null) return new Dictionary<int, int>();

            var refs = new List<ProxyRef>();
            foreach (var p in proxies)
            {
                if (p.NamedSelectionIndex >= 0 && p.NamedSelectionIndex < odolLod.NamedSelections.Length)
                    refs.Add(new ProxyRef(p.NamedSelectionIndex, p.Transformation.Matrix, p.BoneIndex));
                else
                    Terminal.Warning($"Proxy {p.ProxyModel} points to missing selection {p.NamedSelectionIndex}");
            }

            var result = MatchProxyFaces(refs, faces, points, vertexBones);
            var missing = refs.Where(p => !result.ContainsKey(p.Selection)).ToList();
            if (missing.Count == 0) return result;

            var pointList = new List<Point>(points);
            var faceList = new List<Face>(faces);
            var normalList = new List<Vector3P>(normals) { new Vector3P(0f, 1f, 0f) };
            int normalIndex = normalList.Count - 1;
            foreach (var proxy in missing)
            {
                var (origin, up, dir) = ProxyCorners(proxy.Transform);
                int first = pointList.Count;
                pointList.Add(new Point(origin, PointFlags.NONE));
                pointList.Add(new Point(up, PointFlags.NONE));
                pointList.Add(new Point(dir, PointFlags.NONE));
                var verts = new Vertex[4];
                for (int i = 0; i < 3; i++) verts[i] = new Vertex(first + i, normalIndex, 0f, 0f);
                verts[3] = new Vertex(0, 0, 0f, 0f);
                faceList.Add(new Face(3, verts, FaceFlags.DEFAULT, "", ""));
                result[proxy.Selection] = faceList.Count - 1;
            }
            points = pointList.ToArray();
            normals = normalList.ToArray();
            faces = faceList.ToArray();
            return result;
        }

        /// <summary>
        /// Matches proxies to untextured triangles that fit their transform. Each face is used
        /// once. Several proxies can share a transform (e.g. rotors and interior at the origin),
        /// so a first pass only accepts faces skinned to the proxy's own bone.
        /// </summary>
        /// <returns>Selection index to face index; proxies without a match are left out.</returns>
        internal static Dictionary<int, int> MatchProxyFaces(IReadOnlyList<ProxyRef> proxies, Face[] faces, Point[] points, (int Bone, byte Weight)[][] vertexBones)
        {
            var result = new Dictionary<int, int>();
            var candidates = Enumerable.Range(0, faces.Length)
                .Where(i => faces[i].VertexCount == 3 && string.IsNullOrEmpty(faces[i].Texture)).ToList();
            var used = new HashSet<int>();

            foreach (bool requireBone in new[] { true, false })
            {
                foreach (var proxy in proxies.Where(p => !result.ContainsKey(p.Selection)))
                {
                    var (origin, up, dir) = ProxyCorners(proxy.Transform);
                    int face = candidates.FirstOrDefault(i => !used.Contains(i)
                        && TriangleMatches(faces[i], points, origin, up, dir)
                        && (!requireBone || FaceUsesBone(faces[i], vertexBones, proxy.Bone)), -1);
                    if (face < 0) continue;
                    used.Add(face);
                    result[proxy.Selection] = face;
                }
            }
            return result;
        }

        /// <summary>
        /// Corners of the triangle that defines a proxy: the right angle at the origin, a 2 m
        /// leg along the up axis and a 1 m leg along the direction axis. Matrix rows are
        /// aside, up, direction and position.
        /// </summary>
        internal static (Vector3P Origin, Vector3P Up, Vector3P Dir) ProxyCorners(System.Numerics.Matrix4x4 m)
        {
            return (
                new Vector3P(m.M41, m.M42, m.M43),
                new Vector3P(m.M41 + m.M21 * ProxyUpLength, m.M42 + m.M22 * ProxyUpLength, m.M43 + m.M23 * ProxyUpLength),
                new Vector3P(m.M41 + m.M31 * ProxyDirLength, m.M42 + m.M32 * ProxyDirLength, m.M43 + m.M33 * ProxyDirLength));
        }

        /// <summary>True if every expected corner coincides with a corner of the face (any order).</summary>
        internal static bool TriangleMatches(Face face, Point[] points, params Vector3P[] expected)
        {
            const float eps = 0.002f;
            var corners = face.RealVertices.Select(v => points[v.PointIndex]).ToArray();
            return expected.All(e => corners.Any(c =>
                Math.Abs(c.X - e.X) < eps && Math.Abs(c.Y - e.Y) < eps && Math.Abs(c.Z - e.Z) < eps));
        }

        /// <summary>
        /// True if all corners of the face are skinned to the given skeleton bone. A bone of -1
        /// (proxy not attached to a bone) matches faces whose corners are not skinned at all.
        /// </summary>
        internal static bool FaceUsesBone(Face face, (int Bone, byte Weight)[][] vertexBones, int bone)
        {
            return face.RealVertices.All(v =>
            {
                if (v.PointIndex >= vertexBones.Length) return false;
                var bones = vertexBones[v.PointIndex];
                return bone < 0 ? bones.Length == 0 : bones.Any(b => b.Bone == bone);
            });
        }

        private static Vertex MakeMlodVertex(int vertexIndex, Vector3P[] normals, System.Numerics.Vector2[] uvs)
        {
            int normalIdx = vertexIndex;
            if (normals != null && normalIdx >= normals.Length)
                normalIdx = 0;
            return new Vertex(
                vertexIndex,
                normalIdx,
                (uvs != null && vertexIndex < uvs.Length) ? uvs[vertexIndex].X : 0,
                (uvs != null && vertexIndex < uvs.Length) ? uvs[vertexIndex].Y : 0
            );
        }

        private static List<Face> ConvertOneFace(
            Polygon odolFace, Section section, BIS.P3D.ODOL.LOD odolLod,
            System.Numerics.Vector2[] uvs, Vector3P[] normals)
        {
            var result = new List<Face>();
            var vertexCount = odolFace.VertexIndices.Length;

            string texture = (odolLod.Textures != null && section.TextureIndex >= 0 && section.TextureIndex < odolLod.Textures.Length)
                ? odolLod.Textures[section.TextureIndex] : "";
            string material = (odolLod.Materials != null && section.MaterialIndex >= 0 && section.MaterialIndex < odolLod.Materials.Length)
                ? odolLod.Materials[section.MaterialIndex].MaterialName : (section.Material ?? "");

            if (vertexCount <= 4)
            {
                // ODOL order reversed for MLOD winding convention
                var mlodVertices = new Vertex[4];
                for (int i = 0; i < 4; i++)
                {
                    if (i < vertexCount)
                    {
                        mlodVertices[i] = MakeMlodVertex(
                            odolFace.VertexIndices[vertexCount - 1 - i], normals, uvs);
                    }
                    else
                    {
                        mlodVertices[i] = new Vertex(0, 0, 0, 0);
                    }
                }
                result.Add(new Face(vertexCount, mlodVertices, FaceFlags.DEFAULT, texture, material));
            }
            else
            {
                // Fan-triangulate n-gon. ODOL order: [v0, v1, ..., v_{N-1}].
                // MLOD reverses winding: v_{N-1} becomes the fan center.
                // Triangle k: (v_{N-1}, v_{N-2-k}, v_{N-3-k}) for k = 0..N-3
                for (int tri = 0; tri < vertexCount - 2; tri++)
                {
                    var mlodVertices = new Vertex[4];
                    mlodVertices[0] = MakeMlodVertex(
                        odolFace.VertexIndices[vertexCount - 1], normals, uvs);
                    mlodVertices[1] = MakeMlodVertex(
                        odolFace.VertexIndices[vertexCount - 2 - tri], normals, uvs);
                    mlodVertices[2] = MakeMlodVertex(
                        odolFace.VertexIndices[vertexCount - 3 - tri], normals, uvs);
                    mlodVertices[3] = new Vertex(0, 0, 0, 0);

                    result.Add(new Face(3, mlodVertices, FaceFlags.DEFAULT, texture, material));
                }
            }

            return result;
        }
    }
}
